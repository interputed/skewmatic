
#include "TileThreader.hpp"

TileThreader::TileThreader(cvt::Tiler& readTiler, cvt::Tiler& writeTiler, const unsigned int &tileCount, const Parameters &vars)
    : _input(readTiler), _output(writeTiler) , _tileCount(tileCount), _vars(vars)
{

    _tileIndex = 0;
    _percentScaler = _tileCount / 10;
}

void TileThreader::operator()()
{
    while (true) {
        const bool verbose = _vars.verbose;
        const int tileIndex = _tileIndex;
        const int buffer = _vars.buffer;
        ++_tileIndex;

        // Thread stop case.
        if (tileIndex < 0 || tileIndex >= _tileCount) {
            return;
        } else if (verbose) {
            ProgressTracker(tileIndex);
        }

        cvt::cvTile<unsigned short> iTile;

        { // Read Lock Scope.
            std::lock_guard<std::mutex> inLock(_readMutex);

            // Since using buffer, ROI must be reduced before being written to disk.
            iTile = _input.getCvTile<unsigned short>(tileIndex, buffer);
        } // End Read Lock

        // Reduce depth from 11-bit to 8-bit.
        // Can add additional logic in ReduceDepth for more conversions if necessary.
        cvt::cvTile<unsigned char> reducedTile = ReduceDepth(iTile);
        cvt::cvTile<unsigned char> outTile = AlgSequence(reducedTile);
        
        { // Write Lock Scope.
            std::lock_guard<std::mutex> outLock(_writeMutex);
            
            // Removes buffer area before writing.
            outTile.constrictROI(buffer);
            _output.putCvTile(outTile, tileIndex);
        } // End Write Lock

    } // End While

}

cvt::cvTile<unsigned char> TileThreader::AlgSequence(cvt::cvTile<unsigned char> &iTile)
{
    const auto tileSize = iTile.getSize();
    const auto bandCount = iTile.getBandCount();

    // Close filter, then create close mask.
    cvt::cvTile<unsigned char> closeTile(tileSize, bandCount);
    Close(iTile, closeTile);
    cvt::cvTile<unsigned char> outTileClose(tileSize, bandCount);
    Mask(closeTile, outTileClose);

    // Open filter, then create open mask.
    cvt::cvTile<unsigned char> openTile(tileSize, bandCount);
    Open(iTile, openTile);
    cvt::cvTile<unsigned char> outTileOpen(tileSize, bandCount);
    Mask(openTile, outTileOpen);

    // Merge the open and close masks together, return combined mask.
    cvt::cvTile<unsigned char> outTile(tileSize, bandCount);
    Merge(outTileClose, outTileOpen, outTile);
    
    return outTile;
}

void TileThreader::ProgressTracker(const int idx)
{
    if (_tileCount >= 10) {
        if (idx == (_tileCount - 1)) { // Finished
            std::cout << "100%]" << std::endl;
        } else if ((idx % _percentScaler) == 0) { // When tile index is a multiple of the scaler
            if (idx == 0) {
                std::cout << "\nPROCESSING TILES\n[0%  " << std::flush;
            } else {
                std::cout << ((idx / _percentScaler) * 10) << "%  " << std::flush;
            }
        }
    } else { // Just show tile ID for tile counts under 10.
        std::cout << "Tile[" << idx << "] processed." << std::endl;
    }
}

cvt::cvTile<unsigned char> TileThreader::ReduceDepth(cvt::cvTile<unsigned short> &inTile)
{
    cvt::cvTile<unsigned char> outTile(inTile.getSize(), inTile.getBandCount(), 0);
    // compute scaler, 8b max / 10b max
    const double scaler = 255.0 / 2047.0;
    const int bandMax = inTile.getBandCount();
    const int rowMax = inTile.getSize().height;
    const int colMax = inTile.getSize().width;

    for (int band = 0; band < bandMax; ++band) {
        // iterate over pixels and scale from s to d depth
        for (int row = 0; row < rowMax; ++row) {
            for (int col = 0; col < colMax; ++col) {
                outTile[band].at<unsigned char>(row, col) =
                        static_cast<unsigned char>(scaler * inTile[band].at<unsigned short>(row, col));
            }
        }
    }
    return outTile;
}


void TileThreader::Mask(cvt::cvTile<unsigned char> &inTile, cvt::cvTile<unsigned char> &outTile)
{
    const cv::Mat outTileMat = inTile[0];
    cv::Mat gradX, gradY;

    cv::Scharr(outTileMat, gradX, CV_16S, 1, 0);
    cv::Scharr(outTileMat, gradY, CV_16S, 0, 1);

    cv::Mat edgeScaledOut(gradX.size(), CV_8UC1);
    const int rowMax = edgeScaledOut.size().height;
    const int colMax = edgeScaledOut.size().width;

    for (int row = 0; row < rowMax; ++row) {
        for (int col = 0; col < colMax; ++col) {
            const short dx = gradX.at<short>(row, col);
            const short dy = gradY.at<short>(row, col);
            edgeScaledOut.at<unsigned char>(row, col) = cv::saturate_cast<unsigned char>(sqrt(dx * dx + dy * dy));
        }
    }

    // A vector that will store the parameters (x_{start}, y_{start}, x_{end}, y_{end}) of the detected lines.
    std::vector<cv::Vec4i> lines;
    
    cv::HoughLinesP(edgeScaledOut, lines, _vars.rho, _vars.theta, _vars.thresh, _vars.min_length, _vars.max_gap);

    cv::Mat houghMat = cv::Mat::zeros(inTile.getSize().height, inTile.getSize().width, CV_8U);

    // Only write lines if there's enough lines found to likely contain a skew
    if (lines.size() > 120) {
        const unsigned char LINE_WIDTH = 150;
        for (auto &l : lines) {
            // Color of Line, thickness, CV_AA -> antialiased line
            cv::line(houghMat, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), 255, LINE_WIDTH, CV_AA);
        }
    }

    outTile[0] = houghMat;
}


void TileThreader::Open(cvt::cvTile<unsigned char> &inTile, cvt::cvTile<unsigned char> &outTile)
{
    const cv::Point anchor = cv::Point(-1,-1);
    cv::Mat elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(_vars.elem_radius, _vars.elem_radius));

    cv::Mat dilateTile = outTile[0];
	cv::dilate(inTile[0], dilateTile, elem, anchor, _vars.elem_iter);
	cv::Mat erodeTile = dilateTile;
	cv::erode(dilateTile, erodeTile, elem, anchor, _vars.elem_iter);

	outTile[0] = erodeTile;
}


void TileThreader::Close(cvt::cvTile<unsigned char> &inTile, cvt::cvTile<unsigned char> &outTile)
{
    const cv::Point anchor = cv::Point(-1,-1);
    cv::Mat elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(_vars.elem_radius, _vars.elem_radius));

    cv::Mat erodeTile = outTile[0];
    cv::erode(inTile[0], erodeTile, elem, anchor, _vars.elem_iter);
	cv::Mat dilateTile = erodeTile;
	cv::dilate(erodeTile, dilateTile, elem, anchor, _vars.elem_iter);

	outTile[0] = dilateTile;
}

void TileThreader::Merge(cvt::cvTile<unsigned char> &closeTile, cvt::cvTile<unsigned char> &openTile, 
                     cvt::cvTile<unsigned char> &mergeTile)
{
    cv::Mat closed = closeTile[0];
    cv::Mat opened = openTile[0];
    cv::Mat merged = mergeTile[0];

    cv::bitwise_or(closed, opened, merged);
}

std::string InputValidation(const std::string fin)
{
    fs::path input_path(fin);
    if (!fs::exists(input_path)) {
        throw std::invalid_argument(std::string("Invalid image path."));
    }
    return input_path.string();
}


std::string OutputValidation(const std::string fin, const int angle)
{
    fs::path input_path(fin);
    std::string output_file_mod = input_path.filename().string();

    std::stringstream ss;
    ss << "_rotated_" << angle;
    output_file_mod.insert(output_file_mod.length() - input_path.extension().string().length(), ss.str());
    fs::path output_path(output_file_mod);
    // If output_path is a file that already exists, deletes it.
    if (fs::exists(output_path)) {
        // If could not delete already existing file.
        if (!fs::remove(output_path)) {
            throw std::invalid_argument(std::string("Failed to remove existing output file."));
        }
    }
    return output_path.string();
}


std::string OutputValidation(const std::string fin, const std::string append, const int angle)
{
    fs::path input_path(fin);
    std::string output_file_mod = input_path.filename().string();

    if (!angle) {
        output_file_mod.insert(output_file_mod.length() - input_path.extension().string().length(), append);        
    } else {
        std::stringstream ss;
        ss << "_rotated_" << angle << append;
        output_file_mod.insert(output_file_mod.length() - input_path.extension().string().length(), ss.str());        
    }

    fs::path output_path(output_file_mod);
    // If output_path is a file that already exists, deletes it.
    if (fs::exists(output_path)) {
        // If could not delete already existing file.
        if (!fs::remove(output_path)) {
            throw std::invalid_argument(std::string("Failed to remove existing output file."));
        }
    }
    return output_path.string();
}
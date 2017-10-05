#include "TileThreader.hpp"



int main(int argc, char **argv)
{
    try {
        // Program Options
        po::options_description desc("Options");
        desc.add_options()
            ("help,h", "Display usage instructions")
            ("verbose,v", "Verbose output, useful for debugging")
            ("rotate,r", po::value<int>()->default_value(0), "Rotation angle in degrees")
            ("scale", po::value<int>()->default_value(1), "Scalar for rotated image")
            ("input,i", po::value<std::string>()->required(), "Required: Input image filename")
            ("rho", po::value<double>()->default_value(1), "Distance resolution of the accumulator in pixels.")
            ("theta", po::value<double>()->default_value(CV_PI / 90), "Angle resolution of the accumulator in radians.")
            ("thresh", po::value<int>()->default_value(200), "Number of votes required to detect a line.")
            ("min_length", po::value<double>()->default_value(500), "Shorter lines than this are rejected.")
            ("max_gap", po::value<double>()->default_value(1), "Max allowed gap between points on line to link them.")
            ("append_name", po::value<std::string>()->default_value("_mask"), "If output filename not given, inserts this value to end of filename, before the extension.")
            ("tile_dim", po::value<int>()->default_value(1024), "Width of tiles to process within image in pixels.")
            ("buffer", po::value<int>()->default_value(512), "Additional buffer size to process outside of tile boundary.")
            ("elem_radius", po::value<int>()->default_value(17), "Element dimensions for erode and dilate operations.")
            ("elem_iter", po::value<int>()->default_value(1), "Number of times to perform erode & dilate operations.");
            
        // Positional Options
        po::positional_options_description positionalOptions;
        positionalOptions.add("input", 1);
        // Variable to contain all the program arguments
        po::variables_map vm;

        try {
            po::store(po::command_line_parser(argc, argv).options(desc).positional(positionalOptions).run(), vm);
            if (vm.count("help")) {
                std::cout << "Skewmatic" << std::endl << desc << std::endl;
                return SUCCESS;
            }
            po::notify(vm);
        } catch(po::error &e) {
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
            std::cerr << desc << std::endl;
            return ERROR_IN_COMMAND_LINE;
        }

    // MAIN APP LOGIC //
    Parameters vars(vm);
    const bool verbose = vars.verbose;
    const int angle = vars.angle;
    const double scale = vars.scale;

    if (verbose) {
        std::cout << "Input File:  " << vars.input_file << std::endl;
        std::cout << "Output File: " << vars.output_file << std::endl;
    }

    // Rotate image if desired
    // TODO: Figure out how to expand the image so edge data is not lost on rotation.
    if (angle) {
        if (verbose) {
            std::cout << "Rotating image " << angle << "degrees at " << scale << " scale...\n" 
            << "WARNING: Rotation is for test purposes only, some edge data will be lost after rotation!" << std::endl;
        }
        vars.input_file = RotateImage(vars.input_file, angle, scale);
        if (verbose) {
            std::cout << "New Input File (rotated): " << vars.input_file << std::endl;
        }
    }

    // Create and initialize Tiler objects
    cvt::Tiler read_tiler, write_tiler;
    read_tiler.open(vars.input_file);
    const cv::Size2i raster_size(read_tiler.getRasterSize());
    const cv::Size2i tile_size(vars.tile_dim, vars.tile_dim);
    write_tiler.create(vars.output_file, "GTiff", raster_size, 1, cvt::Depth8U);
    read_tiler.setCvTileSize(tile_size);
    write_tiler.setCvTileSize(tile_size);
    write_tiler.copyMetadata(read_tiler);

    // Create threads for tile processing
    const unsigned int thread_limit = (unsigned int)sysconf(_SC_NPROCESSORS_ONLN);
    const unsigned int tile_count = (unsigned int)read_tiler.getCvTileCount();
    std::vector<std::thread> threads;
    threads.reserve(thread_limit);

    // Create a vector of threads which are re-used until all tiles are processed.
    TileThreader threader(read_tiler, write_tiler, tile_count, vars);

    for (unsigned int i = 0; i < std::min(thread_limit, tile_count); ++i) {
        threads.emplace_back(std::thread(std::ref(threader)));
    }

    // Capture and join all threads that make it here.
    for (auto &x : threads) {
        x.join();
    }
    

    } catch(std::exception &e) {
        std::cerr << "Unhandled Exception reached the top of main: " << e.what() << ", application will now exit"
                  << std::endl;
        return ERROR_UNHANDLED_EXCEPTION;
    }

    return SUCCESS;
}


std::string RotateImage(const std::string fin, const int angle, const double scale)
{
    cv::Mat src = cv::imread(fin, CV_16U);
    cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, src.type());
    const cv::Point center = cv::Point(dst.cols / 2, dst.rows / 2);
    const cv::Mat rotation_vector = cv::getRotationMatrix2D(center, angle, scale);
    cv::warpAffine(src, dst, rotation_vector, src.size());
    const std::string new_fin = OutputValidation(fin, angle);
    cv::imwrite(new_fin, dst);
    // src is no longer needed
    src.refcount = 0;
    src.release();

    return new_fin;
}
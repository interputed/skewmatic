#pragma once

#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <cvtile/base/cvTile.hpp>
#include <cvtile/base/Tiler.hpp>


std::string InputValidation(const std::string fin);
std::string OutputValidation(const std::string fin, const int angle);
std::string OutputValidation(const std::string fin, const std::string append, const int angle);
std::string RotateImage(const std::string fin, const int angle, const double scale);


namespace fs = boost::filesystem;
namespace po = boost::program_options;
namespace {
    const size_t SUCCESS = 0;
    const size_t ERROR_IN_COMMAND_LINE = 1;
    const size_t ERROR_UNHANDLED_EXCEPTION = 2;
}


// Simple container class used for all possible arguments.
class Parameters {
  public:
    bool verbose;
    std::string input_file;
    std::string output_file;
    int angle;
    double scale;
    int tile_dim;
    int buffer;
    int elem_radius;
    int elem_iter;

    // Used for morphological filtering
    double rho;
    double theta;
    int thresh;
    double min_length;
    double max_gap;

    Parameters(po::variables_map vm) {
        if (vm.count("verbose")) {
            verbose = true;
        } else {
            verbose = false;
        }
        const std::string temp1(vm["input"].as<std::string>());
        input_file = InputValidation(temp1);
        angle = vm["rotate"].as<int>();
        const std::string temp2(vm["append_name"].as<std::string>());
        output_file = OutputValidation(input_file, temp2, angle);
        scale = vm["scale"].as<double>();
        tile_dim = vm["tile_dim"].as<int>();
        buffer = vm["buffer"].as<int>();
        elem_radius = vm["elem_radius"].as<int>();
        elem_iter = vm["elem_iter"].as<int>();
        rho = vm["rho"].as<double>();
        theta = vm["theta"].as<double>();
        thresh = vm["thresh"].as<int>();
        min_length = vm["min_length"].as<double>();
        max_gap = vm["max_gap"].as<double>();
    }
    
  
};


class TileThreader
{
  public:

    TileThreader(cvt::Tiler &readTiler, cvt::Tiler &writeTiler, const unsigned int &tileCount, const Parameters &vars);

    // Thread loop functor to actually work on nodes
    void operator()();

  private:
    cvt::Tiler _input;
    cvt::Tiler _output;
    std::atomic<int> _tileIndex;
    int _tileCount;

    // Helper for progress check. ~10% of _tileCount;
    int _percentScaler;

    // I/O mutex
    std::mutex _readMutex;
    std::mutex _writeMutex;

    Parameters _vars;

    cvt::cvTile<unsigned char> AlgSequence(cvt::cvTile<unsigned char> &iTile);
    void ProgressTracker(const int idx);
    cvt::cvTile<unsigned char> ReduceDepth(cvt::cvTile<unsigned short> &inTile);
    void Mask(cvt::cvTile<unsigned char> &inTile, cvt::cvTile<unsigned char> &outTile);
    void Open(cvt::cvTile<unsigned char> &inTile, cvt::cvTile<unsigned char> &outTile);
    void Close(cvt::cvTile<unsigned char> &inTile, cvt::cvTile<unsigned char> &outTile);
    void Merge(cvt::cvTile<unsigned char> &closeTile, cvt::cvTile<unsigned char> &openTile, 
        cvt::cvTile<unsigned char> &mergeTile);
};


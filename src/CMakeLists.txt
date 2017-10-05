add_executable(skewmatic skewmatic.cpp)

target_link_libraries(cvtapp
	boost_program_options
    boost_filesystem
    boost_regex
    boost_system
    boost_iostreams
    cvtile
    gdal
    opencv_core
    opencv_imgproc
    opencv_highgui
	pthread
)
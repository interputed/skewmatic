add_executable(skewmatic skewmatic.cpp)

add_library(TileThreader TileThreader.cpp)

target_link_libraries(skewmatic
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
    TileThreader
)

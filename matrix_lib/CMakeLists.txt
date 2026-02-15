set(NAME Matrix)

add_library(${NAME} INTERFACE)

add_library(MatrixLib::Matrix ALIAS ${NAME})

target_include_directories(${NAME} INTERFACE 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
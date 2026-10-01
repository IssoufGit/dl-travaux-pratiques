cmake_minimum_required(VERSION 3.16)
project(beans_classifier LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Génère build/compile_commands.json, utilisé par clangd (voir .clangd et le README)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# --- OpenCV : installé par le gestionnaire de paquets (libopencv-dev) --------
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui)

# --- ONNX Runtime -----------------------------------------------------------
# ONNX Runtime ne fournit pas de paquet CMake officiel : on indique donc le
# dossier obtenu en décompressant l'archive onnxruntime-linux-x64-<version>.tgz
# avec -DONNXRUNTIME_ROOT=/chemin/vers/onnxruntime-linux-x64-1.30.0
set(ONNXRUNTIME_ROOT "" CACHE PATH "Dossier racine d'ONNX Runtime (contient include/ et lib/)")

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}
    PATH_SUFFIXES include include/onnxruntime
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS ${ONNXRUNTIME_ROOT}
    PATH_SUFFIXES lib lib64
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "ONNX Runtime introuvable.\n"
        "Téléchargez l'archive depuis "
        "https://github.com/microsoft/onnxruntime/releases, décompressez-la, "
        "puis relancez CMake en indiquant son emplacement :\n"
        "  cmake -B build -DONNXRUNTIME_ROOT=$HOME/onnxruntime-linux-x64-1.30.0")
endif()

message(STATUS "ONNX Runtime : ${ONNXRUNTIME_LIBRARY}")

# --- Exécutable -------------------------------------------------------------
add_executable(beans_classifier beans_classifier.cpp)

target_include_directories(beans_classifier PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(beans_classifier PRIVATE ${ONNXRUNTIME_LIBRARY} ${OpenCV_LIBS})

# Enregistre le chemin de libonnxruntime.so dans le binaire : il se lance ainsi
# directement, sans avoir à définir LD_LIBRARY_PATH.
get_filename_component(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
set_target_properties(beans_classifier PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
)

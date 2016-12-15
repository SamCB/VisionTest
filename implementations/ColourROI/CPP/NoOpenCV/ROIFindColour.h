#ifndef ROI_FIND_COLOUR_H
#define ROI_FIND_COLOUR_H

#if defined(_MSC_VER)
    //  Microsoft 
    #define EXPORT __declspec(dllexport)
    #define IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
    //  GCC
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
#else
    //  do nothing and hope for the best?
    #define EXPORT
    #define IMPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif

extern "C" int** EXPORT ROIFindColour(float*** baseIm, int* shape, int* numRegions);

#endif  // ROI_FIND_COLOUR_H

#ifndef ROI_FIND_COLOUR_H
#define ROI_FIND_COLOUR_H

#ifdef BUILDING_DLL
#define DLL_SETTING __declspec(dllexport)
#else
#define DLL_SETTING __declspec(dllimport)
#endif

int** DLL_SETTING ROIFindColour(float*** baseIm, int* shape, int* numRegions);

#endif  // ROI_FIND_COLOUR_H
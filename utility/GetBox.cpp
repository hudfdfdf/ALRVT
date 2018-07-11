/**
 * @file resize.cpp
 * @brief mex interface for resize
 * @author Kota Yamaguchi
 * @date 2012
 */
#include "mexopencv.hpp"
#include "bounding_box.h"
#include "image_proc.h"
using namespace std;
using namespace cv;

/**
 * Main entry called from Matlab
 * @param nlhs number of left-hand-side arguments
 * @param plhs pointers to mxArrays in the left-hand-side
 * @param nrhs number of right-hand-side arguments
 * @param prhs pointers to mxArrays in the right-hand-side
 */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
         
    // Argument vector
    vector<MxArray> rhs(prhs,prhs+nrhs);

  Mat image_curr(rhs[0].toMat());
  Mat curr_search_region(rhs[1].toMat());
  BoundingBox bbox_estimate_unscaled;
BoundingBox bbox_estimate_uncentered;
   BoundingBox bbox_estimate;
     double edge_spacing_x=rhs[2].toDouble();
  double edge_spacing_y=rhs[3].toDouble(); 
  BoundingBox search_location;
  search_location.x1_=rhs[4].toDouble();
  search_location.y1_=rhs[5].toDouble();
  search_location.x2_=rhs[6].toDouble();
  search_location.y2_=rhs[7].toDouble();
  bbox_estimate.x1_=rhs[8].toDouble();
  bbox_estimate.y1_=rhs[9].toDouble();
  bbox_estimate.x2_=rhs[10].toDouble();
  bbox_estimate.y2_=rhs[11].toDouble();
   
  bbox_estimate.Unscale(curr_search_region, &bbox_estimate_unscaled);

  // Find the estimated bounding box location relative to the current crop.
  bbox_estimate_unscaled.Uncenter(image_curr, search_location, edge_spacing_x, edge_spacing_y, &bbox_estimate_uncentered);
  plhs[0]=MxArray(bbox_estimate_uncentered.x1_);
  plhs[1]=MxArray(bbox_estimate_uncentered.y1_);
  plhs[2]=MxArray(bbox_estimate_uncentered.x2_);
  plhs[3]=MxArray(bbox_estimate_uncentered.y2_);
 
}

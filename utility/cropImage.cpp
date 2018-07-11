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
    // Check the number of arguments
    
    // Argument vector
    vector<MxArray> rhs(prhs,prhs+nrhs);

  // Crop the current image based on predicted prior location of target.
  Mat image_curr(rhs[0].toMat()),curr_search_region;
  BoundingBox bbox_curr_prior_tight_;
  bbox_curr_prior_tight_.x1_=rhs[1].toDouble();
  bbox_curr_prior_tight_.y1_=rhs[2].toDouble();
  bbox_curr_prior_tight_.x2_=rhs[3].toDouble();
  bbox_curr_prior_tight_.y2_=rhs[4].toDouble();
  BoundingBox search_location;
  double edge_spacing_x, edge_spacing_y;
  CropPadImage(bbox_curr_prior_tight_, image_curr, &curr_search_region, &search_location, &edge_spacing_x, &edge_spacing_y);
  plhs[0]=MxArray(curr_search_region); 
 plhs[1]=MxArray(edge_spacing_x);
  plhs[2]=MxArray(edge_spacing_y);
  plhs[3]=MxArray(search_location.x1_);
  plhs[4]=MxArray(search_location.y1_);
  plhs[5]=MxArray(search_location.x2_);
  plhs[6]=MxArray(search_location.y2_);
//  plhs[3]=MxArray(search_location);
 
}

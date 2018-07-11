#include "bounding_box.h"

#include <cstdio>

// Uncomment line below if you want to use rectangles

#include "helper.h"

// How much context to pad the image and target with (relative to the
// bounding box size).
const double kContextFactor = 2;

// Factor by which to scale the bounding box coordinates, based on the
// neural network default output range.
const double kScaleFactor = 10;

// If true, the neural network will estimate the bounding box corners: (x1, y1, x2, y2)
// If false, the neural network will estimate the bounding box center location and size: (center_x, center_y, width, height)
const bool use_coordinates_output = true;

BoundingBox::BoundingBox() :
  scale_factor_(kScaleFactor)
{
}


BoundingBox::BoundingBox(const std::vector<float>& bounding_box)
  : scale_factor_(kScaleFactor)
{
  if (bounding_box.size() != 4) {
    printf("Error - bounding box vector has %zu elements\n",
           bounding_box.size());
  }

  if (use_coordinates_output) {
    // Set bounding box coordinates.
    x1_ = bounding_box[0];
    y1_ = bounding_box[1];
    x2_ = bounding_box[2];
    y2_ = bounding_box[3];
  } else {
    // Get bounding box in format: (center_x, center_y, width, height)
    const double center_x = bounding_box[0];
    const double center_y = bounding_box[1];
    const double width = bounding_box[2];
    const double height = bounding_box[3];

    // Convert (center_x, center_y, width, height) to (x1, y1, x2, y2).
    x1_ = center_x - width / 2;
    y1_ = center_y - height / 2;
    x2_ = center_x + width / 2;
    y2_ = center_y + height / 2;
  }
}

void BoundingBox::GetVector(std::vector<float>* bounding_box) const {
  if (use_coordinates_output) {
    // Convert bounding box into a vector format using (x1, y1, x2, y2).
    bounding_box->push_back(x1_);
    bounding_box->push_back(y1_);
    bounding_box->push_back(x2_);
    bounding_box->push_back(y2_);
  } else {
    // Convert bounding box into a vector format using (center_x, center_y, width, height).
    bounding_box->push_back(get_center_x());
    bounding_box->push_back(get_center_y());
    bounding_box->push_back(get_width());
    bounding_box->push_back(get_height());
  }
}

void BoundingBox::Print() const {
  printf("Bounding box: x,y: %lf, %lf, %lf, %lf, w,h: %lf, %lf\n", x1_, y1_, x2_, y2_, get_width(), get_height());
}

void BoundingBox::Scale(const cv::Mat& image, BoundingBox* bbox_scaled) const {
  *bbox_scaled = *this;

  const int width = image.cols;
  const int height = image.rows;

  // Scale the bounding box so that the coordinates range from 0 to 1.
  bbox_scaled->x1_ /= width;
  bbox_scaled->y1_ /= height;
  bbox_scaled->x2_ /= width;
  bbox_scaled->y2_ /= height;

  // Scale the bounding box so that the coordinates range from 0 to scale_factor_.
  bbox_scaled->x1_ *= scale_factor_;
  bbox_scaled->x2_ *= scale_factor_;
  bbox_scaled->y1_ *= scale_factor_;
  bbox_scaled->y2_ *= scale_factor_;
}

void BoundingBox::Unscale(const cv::Mat& image, BoundingBox* bbox_unscaled) const {
  *bbox_unscaled = *this;

  const int image_width = image.cols;
  const int image_height = image.rows;

  // Unscale the bounding box so that the coordinates range from 0 to 1.
  bbox_unscaled->x1_ /= scale_factor_;
  bbox_unscaled->x2_ /= scale_factor_;
  bbox_unscaled->y1_ /= scale_factor_;
  bbox_unscaled->y2_ /= scale_factor_;

  // Unscale the bounding box so that the coordinates match the original image coordinates
  // (undoing the effect from the Scale method).
  bbox_unscaled->x1_ *= image_width;
  bbox_unscaled->y1_ *= image_height;
  bbox_unscaled->x2_ *= image_width;
  bbox_unscaled->y2_ *= image_height;
}

double BoundingBox::compute_output_width() const {
  // Get the bounding box width.
  const double bbox_width = (x2_ - x1_);

  // We pad the image by a factor of kContextFactor around the bounding box
  // to include some image context.
  const double output_width = kContextFactor * bbox_width;

  // Ensure that the output width is at least 1 pixel.
  return std::max(1.0, output_width);
}

double BoundingBox::compute_output_height() const {
  // Get the bounding box height.
  const double bbox_height = (y2_ - y1_);

  // We pad the image by a factor of kContextFactor around the bounding box
  // to include some image context.
  const double output_height = kContextFactor * bbox_height;

  // Ensure that the output height is at least 1 pixel.
  return std::max(1.0, output_height);
}

double BoundingBox::get_center_x() const {
  // Compute the bounding box center x-coordinate.
  return (x1_ + x2_) / 2;
}

double BoundingBox::get_center_y() const {
  // Compute the bounding box center y-coordinate.
  return (y1_ + y2_) / 2;
}

void BoundingBox::Recenter(const BoundingBox& search_location,
              const double edge_spacing_x, const double edge_spacing_y,
              BoundingBox* bbox_gt_recentered) const {
  // Location of bounding box relative to the focused image and edge_spacing.
  bbox_gt_recentered->x1_ = x1_ - search_location.x1_ + edge_spacing_x;
  bbox_gt_recentered->y1_ = y1_ - search_location.y1_ + edge_spacing_y;
  bbox_gt_recentered->x2_ = x2_ - search_location.x1_ + edge_spacing_x;
  bbox_gt_recentered->y2_ = y2_ - search_location.y1_ + edge_spacing_y;
}

void BoundingBox::Uncenter(const cv::Mat& raw_image,
                           const BoundingBox& search_location,
                           const double edge_spacing_x, const double edge_spacing_y,
                           BoundingBox* bbox_uncentered) const {
  // Undo the effect of Recenter.
  bbox_uncentered->x1_ = std::max(0.0, x1_ + search_location.x1_ - edge_spacing_x);
  bbox_uncentered->y1_ = std::max(0.0, y1_ + search_location.y1_ - edge_spacing_y);
  bbox_uncentered->x2_ = std::min(static_cast<double>(raw_image.cols), x2_ + search_location.x1_ - edge_spacing_x);
  bbox_uncentered->y2_ = std::min(static_cast<double>(raw_image.rows), y2_ + search_location.y1_ - edge_spacing_y);
}

double BoundingBox::edge_spacing_x() const {
  const double output_width = compute_output_width();
  const double bbox_center_x = get_center_x();

  // Compute the amount that the output "sticks out" beyond the edge of the image (edge effects).
  // If there are no edge effects, we would have output_width / 2 < bbox_center_x, but if the crop is near the left
  // edge of the image then we would have output_width / 2 > bbox_center_x, with the difference
  // being the amount that the output "sticks out" beyond the edge of the image.
  return std::max(0.0, output_width / 2 - bbox_center_x);
}

double BoundingBox::edge_spacing_y() const {
  const double output_height = compute_output_height();
  const double bbox_center_y = get_center_y();

  // Compute the amount that the output "sticks out" beyond the edge of the image (edge effects).
  // If there are no edge effects, we would have output_height / 2 < bbox_center_y, but if the crop is near the bottom
  // edge of the image then we would have output_height / 2 > bbox_center_y, with the difference
  // being the amount that the output "sticks out" beyond the edge of the image.
  return std::max(0.0, output_height / 2 - bbox_center_y);
}

void BoundingBox::Draw(const int r, const int g, const int b,
                       cv::Mat* image) const {
  // Get the top-left point.
  const cv::Point point1(x1_, y1_);

  // Get the bottom-rigth point.
  const cv::Point point2(x2_, y2_);

  // Get the selected color.
  const cv::Scalar box_color(b, g, r);

  // Draw a rectangle corresponding to this bbox with the given color.
  const int thickness = 3;
  cv::rectangle(*image, point1, point2, box_color, thickness);
}

void BoundingBox::DrawBoundingBox(cv::Mat* image) const {
  // Draw a white bounding box on the image.
  Draw(255, 255, 255, image);
}



double BoundingBox::compute_intersection(const BoundingBox& bbox) const {
  const double area = std::max(0.0, std::min(x2_, bbox.x2_) - std::max(x1_, bbox.x1_)) * std::max(0.0, std::min(y2_, bbox.y2_) - std::max(y1_, bbox.y1_));
  return area;
}

double BoundingBox::compute_area() const {
  return get_width() * get_height();
}

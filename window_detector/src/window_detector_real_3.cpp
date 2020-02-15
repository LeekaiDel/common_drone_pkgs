#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <stereo_msgs/DisparityImage.h>
#include <cv_bridge/cv_bridge.h>
#include <stdio.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>

#include "window_detector_debug.h"
#include <drone_msgs/WindowAngleDir.h>

using namespace cv;
using namespace std;

ros::Publisher window_dir_publisher;

cv_bridge::CvImagePtr raw_image_cv_ptr = NULL;
cv_bridge::CvImagePtr res_image_cv_ptr = NULL;
//cv_bridge::CvImagePtr disp_image_cv_ptr = NULL;
stereo_msgs::DisparityImage disp_img;

cv_bridge::CvImagePtr disp_image_cv_ptr = NULL;
cv::Mat res_image_test;

bool received_new_image = true;
bool received_new_disp_image = false;

struct gradLine
{
  cv::Point start;
  cv::Point end;
  cv::Point center;
  float line_ang;
  float grad_ang;
};

struct gradRect
{
  std::vector<gradLine> lines;
  cv::Point center;
};

/*
 * Convert ROS disparity msg to opencv CV_8UC1 image.
 */
cv::Mat getDispImg(stereo_msgs::DisparityImage disp, cv::Rect disp_roi)
{
//  cv::Rect disp_roi(75, 10, 555, 460);
  cv::Mat disp_img(disp_image_cv_ptr->image, disp_roi);
  cv::Mat img(disp_img.rows, disp_img.cols, CV_8UC1);
  for(int i = 0; i < disp_img.rows; i++)
  {
    float* Di = disp_img.ptr<float>(i);
    char* Ii = img.ptr<char>(i);
    for(int j = 0; j < disp_img.cols; j++)
    {
      Ii[j] = static_cast<char>(255*((Di[j]-disp.min_disparity)/(disp.max_disparity-disp.min_disparity)));
    }
  }
  return img;
}

/*
 * Morphological operations for removing noise.
 * It can remove boundaries or other important data.
 * Use with caution!
 */
void morphTransform(cv::Mat& src, cv::Mat& dst, int k)
{
  // Morphological opening (remove small objects from the foreground)
  cv::erode(src, dst, getStructuringElement(cv::MORPH_RECT, cv::Size(k+1,k+1)));
  cv::dilate(dst, dst, getStructuringElement(cv::MORPH_RECT, cv::Size(k, k)));
  // Morphological closing (fill small holes in the foreground)
  cv::dilate(dst, dst, getStructuringElement(cv::MORPH_RECT, cv::Size(k, k)));
  cv::erode(dst, dst, getStructuringElement(cv::MORPH_RECT, cv::Size(k, k)));
}

/*
 * Return gradient of src image as magnitude and angles matrixes
 */
void gradient(cv::Mat& src, cv::Mat& magnitude, cv::Mat& angles)
{
  // Convert src image from CV_8UC1 to CV_32F
  src.convertTo(src, CV_32F, 1.f/255);

  // Blur image for getting more uniform gradient
  cv::GaussianBlur(src, src, cv::Size(11,11), src.cols*src.rows*0.5);

  // morphTransform(smoothed_plane, smoothed_plane, 20);

  // Get gradient for x and y axes
  cv::Mat grad_x, grad_y;
  cv::Scharr(src, grad_x, CV_32FC1, 1, 0, 5);
  cv::Scharr(src, grad_y, CV_32FC1, 0, 1, 5);

  // Get magnitude and direction by x and y gradients
  cv::cartToPolar(grad_x, grad_y, magnitude, angles);
}

inline double det(double a, double b, double c, double d)
{
  return a*d - b*c;
}

/*
 * Find point of line intersection
 */
bool linesIntersection(cv::Vec4f l1, cv::Vec4f l2, cv::Point2f &p)
{
  p.x = 0;
  p.y = 0;

  double det_l1 = det(l1[0], l1[1], l1[2], l1[3]);
  double det_l2 = det(l2[0], l2[1], l2[2], l2[3]);
  double x1_x2 = l1[0] - l1[2];
  double x3_x4 = l2[0] - l2[2];
  double y1_y2 = l1[1] - l1[3];
  double y3_y4 = l2[1] - l2[3];

  double x_nom = det(det_l1, x1_x2, det_l2, x3_x4);
  double y_nom = det(det_l1, y1_y2, det_l2, y3_y4);
  double de_nom = det(x1_x2, y1_y2, x3_x4, y3_y4);

  if(fabs(de_nom) < 1e-6)  // Lines don't seem to cross
    return false;

  p.x = x_nom / de_nom;
  p.y = y_nom / de_nom;

  if(!isfinite(p.x) || !isfinite(p.y)) //Probably a numerical issue
    return false;

  return true; //All OK
}

/*
 * Normalize angle in range [-pi..pi]
 */
double norm_ang(double ang)
{
  while(ang < -M_PI)
    ang += M_PI*2;
  while(ang > M_PI)
    ang -= M_PI*2;
  return ang;
}

/*
 * Check receiving new image
 */
bool hasImage()
{
  if (disp_image_cv_ptr == NULL)
    return false;
  if(raw_image_cv_ptr == NULL)
    return false;
  if (!received_new_image || !received_new_disp_image)
    return false;

  return true;
}

/*
 * Bilateral filter with kernel (adjusted)
 */
void myBilateralFilter(cv::Mat& img)
{
  cv::Mat tes_bil;
  int filt_kernel = 24;
  cv::bilateralFilter(img, tes_bil, filt_kernel, filt_kernel*2, filt_kernel/2);
  tes_bil.copyTo(img);
}

/*
 * Remove distance transform mask image from image
 */
void removeNoiseByDistTransform(cv::Mat& img)
{
  cv::Mat dist_img;
  img.copyTo(dist_img);
  cv::distanceTransform(img, dist_img, CV_DIST_FAIR, 0);
  cv::normalize(dist_img, dist_img, 0, 1., cv::NORM_MINMAX);
  dist_img *= 0.7;

  cv::Mat dist_img_ch;
  dist_img.convertTo(dist_img_ch, CV_8UC1, 255);
  img-=dist_img_ch;
  img *= 2;
}

/*
 * Remove all contours with area < max_area
 */
void removeSmallContours(std::vector<std::vector<cv::Point>> &cnts, int max_area)
{
  std::vector<std::vector<cv::Point>> filt_cnts;
  for(std::vector<cv::Point> cnt : cnts)
  {
    if (cv::contourArea(cnt) > max_area)
      filt_cnts.push_back(cnt);
  }
  cnts = filt_cnts;
}

/*
 * Aproximate countours
 */
void aproxContours(std::vector<std::vector<cv::Point>> &cnts)
{
  std::vector<std::vector<cv::Point>> approx_cnts(cnts.size());
  std::vector<std::vector<cv::Point>> hull_cnts(cnts.size());
  for(size_t i = 0; i < cnts.size(); i++)
  {
    cv::convexHull( cv::Mat(cnts[i]), hull_cnts[i], false );
    cv::approxPolyDP(cv::Mat(hull_cnts[i]), approx_cnts[i], arcLength(cv::Mat(hull_cnts[i]), true) * 0.07, true);
  }
  cnts = approx_cnts;
}

/*
 * Filter contours by corners count (need 4)
 */
void filtContours(std::vector<std::vector<cv::Point>> &cnts)
{
  std::vector<std::vector<cv::Point>> cnts_filt;
  for(std::vector<cv::Point> cnt : cnts)
  {
    if(cnt.size() == 4)
      cnts_filt.push_back(cnt);
  }
  cnts = cnts_filt;
}

/*
 * Publish direction to window center
 */
void pubResult(const cv::Mat& img, cv::Point window_center)
{
  drone_msgs::WindowAngleDir msg;
  msg.found_window = true;
  msg.width_angle = static_cast<float>(window_center.x) / static_cast<float>(img.cols);
  msg.height_angle = static_cast<float>(window_center.y) / static_cast<float>(img.rows);

  window_dir_publisher.publish(msg);
}

/*
 * Draw result window as red rectangle with yellow circle in center
 */
void drawResult(const cv::Mat &img, gradRect window)
{
  cv::Scalar color = cv::Scalar(0,0,255);
  cv::line(img, window.lines.at(0).start, window.lines.at(2).start, color, 2);
  cv::line(img, window.lines.at(1).start, window.lines.at(3).start, color, 2);

  cv::line(img, window.lines.at(0).start, window.lines.at(1).start, color, 2);
  cv::line(img, window.lines.at(1).start, window.lines.at(2).start, color, 2);
  cv::line(img, window.lines.at(2).start, window.lines.at(3).start, color, 2);
  cv::line(img, window.lines.at(3).start, window.lines.at(0).start, color, 2);

  cv::circle(img, window.center, 10, cv::Scalar(0,255,255),CV_FILLED, 8,0);

  cv::imshow("Result", img); cv::waitKey(1);
}

/*
 * Main process image function
 */
void processImage()
{
  if(!hasImage())
    return;

  // Get disparity map image
  cv::Rect disp_roi(75, 10, 555, 460); // ROI for 640x480 image
  cv::Mat d_img = getDispImg(disp_img, disp_roi);

  // Resize disparity image. Small size - less calculation
  cv::Size disp_size(static_cast<int>(d_img.cols*0.75), static_cast<int>(d_img.rows*0.75));
  cv::resize(d_img, d_img, disp_size, 0, 0, CV_INTER_LINEAR);

  // Create result image for demontrations
  cv::Mat res_img;
  raw_image_cv_ptr->image(disp_roi).copyTo(res_img);
  res_image_test = cv::Mat(res_img.size(), CV_8UC3);
  cv::cvtColor(res_img, res_image_test, CV_GRAY2RGB);
  cv::resize(res_image_test, res_image_test, disp_size, 0, 0, CV_INTER_LINEAR);

  // Find gradient of disparity map image
  cv::Mat magnitude, angles;
  gradient(d_img, magnitude, angles);

  // DEBUG: Draw gradient in new window
  //drawGradient(d_img, magnitude, angles);

  // Convert magnitude to char magnitude_img
  cv::Mat magnitude_img;
  magnitude.convertTo(magnitude_img, CV_8UC1, 255);

  // Blur magnitude
  myBilateralFilter(magnitude_img);

  // DEBUG: Show magnitude
  //cv::imshow("Magnitude", magnitude_img); cv::waitKey(1);

  // Remove noise
  removeNoiseByDistTransform(magnitude_img);

  // DEBUG: Show magnitude after dist transform
  //cv::imshow("Magnitude after dist transform", magnitude_img); cv::waitKey(1);

  // Create result magnitude image for demontrations
  cv::Mat magnitude_res_img = cv::Mat(magnitude_img.size(), CV_8UC3);
  cv::cvtColor(magnitude_img, magnitude_res_img, CV_GRAY2RGB);

  // Find countours in magnitude image
  std::vector<std::vector<cv::Point>> cnts;
  cv::findContours(magnitude_img, cnts, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

  // Remove small countours
  removeSmallContours(cnts, 500);

  // Aproximate contours
  aproxContours(cnts);

  // DEBUG: draw countours
  //drawMyContours(magnitude_res_img, cnts);

  // Remove all contours which doesn't have 4 corner
  filtContours(cnts);

  // Combine the contours and gradient:
  // for each contour get gradient, if gradient has needed direction, then it's part of window
  std::vector<gradRect> raw_windows;
  for(auto cnt : cnts)
  {
    gradRect grad_rect;
    std::vector<int> indexes = {0,1,2,3,0};
    for(size_t i = 0; i < indexes.size()-1; i++)
    {
      gradLine grad_line;
      grad_line.start = cnt.at(indexes[i]);
      grad_line.end = cnt.at(indexes[i+1]);

      grad_line.center.x = (grad_line.start.x + grad_line.end.x) / 2;
      grad_line.center.y = (grad_line.start.y + grad_line.end.y) / 2;

      std::vector<float> line_grad_angles;
      LineIterator it(magnitude_res_img, grad_line.start, grad_line.end, 8);
      float sum = 0;
      for(int j=0; j<it.count; j+=1)
      {
        if(j > 40 && j < it.count-40)
        {
          cv::Point p = it.pos();
          float ang = angles.at<float>(p.y,p.x);
          line_grad_angles.push_back(ang);
          sum += ang;

          cv::Point grad_vec_end;
          grad_vec_end.x = p.x + 20*cos(ang);
          grad_vec_end.y = p.y + 20*sin(ang);
          cv::arrowedLine(magnitude_res_img, p, grad_vec_end, cv::Scalar(0,0,255), 1, 8, 0, 0.4);
        }

        it++;
      }

      //float grad_angles_sum = std::accumulate(line_grad_angles.begin(), line_grad_angles.end(), 0.0f);
      grad_line.grad_ang = sum / line_grad_angles.size();
      if(isnan(grad_line.grad_ang) || isnan(-grad_line.grad_ang))
        grad_line.grad_ang = 0;
      grad_line.line_ang = static_cast<float>(atan2( grad_line.end.y - grad_line.start.y,
                                                     grad_line.end.x - grad_line.start.x ));

      grad_rect.lines.push_back(grad_line);
    }
    cv::Moments mu = cv::moments(cnt);
    grad_rect.center = cv::Point(mu.m10/mu.m00 , mu.m01/mu.m00);

    raw_windows.push_back(grad_rect);
  }

  // Find window (rectangle) among all figures
  cv::RNG rng(12345);
  int w = 0;
  for(auto window : raw_windows)
  {
    cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
    bool true_window = true;
    int l_n = 0;
    for(auto l : window.lines)
    {
      cv::line(magnitude_res_img, l.start, l.end, color, 2);
      cv::Point grad_vec_end;
      //std::cout << w << "-" << l_n << ": grad_ang = " << l.grad_ang << std::endl;
      grad_vec_end.x = l.center.x + 30*cos(l.grad_ang);
      grad_vec_end.y = l.center.y + 30*sin(l.grad_ang);
      cv::arrowedLine(magnitude_res_img, l.center, grad_vec_end, color, 2, 8, 0, 0.4);
      float ang = l.line_ang - l.grad_ang;
      ang = norm_ang(ang)*180/M_PI;
      cv::putText(magnitude_res_img, std::to_string(int(ang)), grad_vec_end,
          cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, cv::Scalar(255,0,0), 1, CV_AA);
      if(ang < 5 || ang > 175 || isnan(ang) || isnan(-ang))
        true_window = false;

      //std::cout << w << "-" << l_n << ": ang = " << ang << " is " << true_window << std::endl;
      l_n ++;
    }
    if(true_window)
    {
      // DEBUG: draw result
      //drawResult(res_image_test, window);
      pubResult(res_image_test, window.center);
    }
    w++;
  }
}

/*
 * Publish result image (it's must be turn off in board version (by default it's turn off))
 */
void pubResImage(const ros::Publisher& pub)
{
  if (disp_image_cv_ptr == NULL)
    return;
  if (!received_new_disp_image)
    return;
  disp_image_cv_ptr->image = res_image_test;
  disp_image_cv_ptr->encoding = "bgr8";
  pub.publish(disp_image_cv_ptr->toImageMsg());

  received_new_image = false;
  received_new_disp_image = false;
}

/*
 * Raw image callback (it can be off if you don't need display or publish image with drawing result)
 */
void rawImageCallback(const sensor_msgs::Image::ConstPtr& msg)
{
  try
  {
    raw_image_cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  received_new_image = true;
}

/*
 * Disparity callback
 */
void disparityImageCallback(const stereo_msgs::DisparityImage::ConstPtr& msg)
{
  disp_img = *msg;
  disp_image_cv_ptr = cv_bridge::toCvCopy(msg->image, sensor_msgs::image_encodings::TYPE_32FC1);
  received_new_disp_image = true;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "window_detector_real_3");

  ros::NodeHandle nh;

  ros::Subscriber raw_image_sub = nh.subscribe("/elp/left/image_raw", 1000, rawImageCallback);
  ros::Subscriber disparity_image_sub = nh.subscribe("/elp/disparity", 1000, disparityImageCallback);
  // ros::Publisher  res_image_pub = nh.advertise<sensor_msgs::Image>("/window_detector_image", 1);
  window_dir_publisher = nh.advertise<drone_msgs::WindowAngleDir>("window_detector/angle_dir", 1);


  ros::Rate loop_rate(10);
  while(ros::ok())
  {
    processImage();

    // pubResImage(res_image_pub);

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}



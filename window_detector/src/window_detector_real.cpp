#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <stereo_msgs/DisparityImage.h>
#include <cv_bridge/cv_bridge.h>
#include <stdio.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>


cv_bridge::CvImagePtr raw_image_cv_ptr = NULL;
cv_bridge::CvImagePtr res_image_cv_ptr = NULL;
//cv_bridge::CvImagePtr disp_image_cv_ptr = NULL;
stereo_msgs::DisparityImage disp_img;

cv_bridge::CvImagePtr disp_image_cv_ptr = NULL;
cv::Mat res_image_test;






bool received_new_image = true;
bool received_new_disp_image = false;


void debug_found_grey_value(cv::Mat img)
{
  std::cout << "DEBUG WIRITTER" << std::endl;
  for (int i = 0; i < 255; i++)
  {
    cv::Mat bin_img;
    cv::threshold(img, bin_img, i, 255, cv::THRESH_BINARY);
    std::string name = "/home/gek/drone_cup/thirdparty/img/img_" +  std::to_string(i) + ".png";
    cv::imwrite(name, bin_img);
    std::cout << "write " << name << std::endl;
  }
  exit(0);
}

void debug_found_hough_val(cv::Mat img, cv::Mat src)
{
  for(int i = 0; i < 255; i++)
  {
    cv::Mat img_i = img.clone();
    std::vector<cv::Vec2f> full_lines;
    cv::HoughLines(src, full_lines, 1, CV_PI/180, i, 0);

    std ::vector< cv::Vec2f >::const_iterator it = full_lines.begin ();
    while ( it != full_lines .end())
    {

      float rho = (*it )[0];                 // first element is distance rho
      float theta = (*it )[1]; // second element is angle theta

      if ( theta < M_PI /4. || theta > 3.* M_PI/4. )
      {     // ~vertical line

        // point of intersection of the line with first row
        cv ::Point pt1( rho / cos (theta), 0);
        // point of intersection of the line with last row
        cv ::Point pt2(( rho - img_i.rows * sin(theta )) / cos(theta ), img_i.rows);

        // draw a while line
        cv ::line( img_i, pt1 , pt2, cv::Scalar (0, 255, 255), 1);
      }
      else
      {    //~horizontal line
        // point of intersection of the line with first column
        cv ::Point pt1( 0, rho / sin( theta));
        // point of intersection of the line with last column
        cv ::Point pt2( img_i.cols , ( rho - img_i.cols * cos(theta )) / sin(theta ));
        // draw a white line
        cv::line( img_i, pt1 , pt2, cv::Scalar (0, 255, 255), 1);
      }
      ++it;
    }
    std::string name = "/home/gek/drone_cup/thirdparty/img/img_" +  std::to_string(i) + ".png";
    cv::imwrite(name, img_i);
    std::cout << "write " << name << std::endl;
  }

  exit(0);
}

void debug_found_hough_p_val(cv::Mat img, cv::Mat src)
{
  for(int i = 0; i < 255; i++)
  {
    cv::Mat img_i = img.clone();
    std::vector<cv::Vec4i> full_lines;
    cv::HoughLinesP(src, full_lines, 1, CV_PI/180, i, 25, 20);

    for( size_t i = 0; i < full_lines.size(); i++ )
    {
      cv::Vec4i l = full_lines[i];
      line(img_i, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,255,255), 2, CV_AA);
    }

    std::string name = "/home/gek/drone_cup/thirdparty/img/img_" +  std::to_string(i) + ".png";
    cv::imwrite(name, img_i);
    std::cout << "write " << name << std::endl;
  }

  exit(0);
}

cv::Mat getDispImg(stereo_msgs::DisparityImage disp)
{
  cv::Rect disp_roi(75, 10, 555, 460);
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

void morphTransform(cv::Mat& src, cv::Mat& dst, int k)
{
  // Morphological opening (remove small objects from the foreground)
  cv::erode(src, dst, getStructuringElement(cv::MORPH_RECT, cv::Size(k+1,k+1)));
  cv::dilate(dst, dst, getStructuringElement(cv::MORPH_RECT, cv::Size(k, k)));
  // Morphological closing (fill small holes in the foreground)
  cv::dilate(dst, dst, getStructuringElement(cv::MORPH_RECT, cv::Size(k, k)));
  cv::erode(dst, dst, getStructuringElement(cv::MORPH_RECT, cv::Size(k, k)));
}

void gradient(cv::Mat& src, cv::Mat& magnitude, cv::Mat& angles)
{
  cv::Mat smoothed_plane = cv::Mat::zeros(src.rows,src.cols, CV_32FC1);
  cv::GaussianBlur(src, smoothed_plane, cv::Size(21,21), src.cols*src.rows*0.5);

  // morphTransform(smoothed_plane, smoothed_plane, 20);

  cv::Mat grad_x, grad_y;
  cv::Scharr(smoothed_plane, grad_x, CV_32FC1, 1, 0, 1);
  cv::Scharr(smoothed_plane, grad_y, CV_32FC1, 0, 1, 1);

  cv::cartToPolar(grad_x, grad_y, magnitude, angles);
}

void drawGradient(cv::Mat &img, cv::Mat& magnitude, cv::Mat& angles, int flow_resolution=10)
{
  if(img.rows != magnitude.rows || img.cols != magnitude.cols ||
     img.rows != angles.rows || img.cols != angles.cols ||
     magnitude.rows != angles.rows || magnitude.cols != angles.cols)
    return;

  for (int i = 0 ; i < img.rows ; i += flow_resolution)
    for (int j = 0 ; j < img.cols ; j+= flow_resolution)
    {
      if(magnitude.at<float>(i,j) < 0.001)
        continue;
      cv::Point2f p(j,i);
      cv::Point2f p2(p.x + 100*magnitude.at<float>(i,j)*cos(angles.at<float>(i,j)),
                     p.y + 100*magnitude.at<float>(i,j)*sin(angles.at<float>(i,j)));

      cv::arrowedLine(img, p, p2, cv::Scalar(0,0,255), 2, 8, 0, 0.4);
    }
}

void drawVectors(cv::Mat &img, std::vector<float> &magnitude, std::vector<float> &angles, std::vector<cv::Point> &coords)
{
  for(int i = 0; i < coords.size(); i++)
  {
    cv::Point p = coords.at(i);
    cv::Point2f p2(p.x + 100*magnitude.at(i)*cos(angles.at(i)),
                   p.y + 100*magnitude.at(i)*sin(angles.at(i)));
    cv::arrowedLine(img, p, p2, cv::Scalar(0,0,255), 2, 8, 0, 0.4);
  }
}

void mergeLinesAndVectors(std::vector<cv::Vec4i> &raw_lines, cv::Mat &raw_ang,
                          cv::Mat &res_ang)
{
  cv::Mat mask = cv::Mat::zeros(raw_ang.size(), CV_8UC1);
  res_ang = cv::Mat::zeros(raw_ang.size(), raw_ang.type());
  for( size_t i = 0; i < raw_lines.size(); i++ )
  {
    cv::Vec4i l = raw_lines[i];
    line(mask, cv::Point(raw_lines[i][0], raw_lines[i][1]), cv::Point(l[2], l[3]), cv::Scalar(255,0,0),3, CV_AA);
  }

  raw_ang.copyTo(res_ang, mask);
}

bool pointOnLine2D (cv::Point p, cv::Point a, cv::Point b, float t = 1E-03f)
{
  // ensure points are collinear
  double zero = (b.x - a.x) * (p.y - a.y) - (p.x - a.x) * (b.y - a.y);
  if (zero > t || zero < -t) return false;

  // check if x-coordinates are not equal
  if (a.x - b.x > t || b.x - a.x > t)
    // ensure x is between a.x & b.x (use tolerance)
    return a.x > b.x
        ? p.x + t > b.x && p.x - t < a.x
        : p.x + t > a.x && p.x - t < b.x;

  // ensure y is between a.y & b.y (use tolerance)
  return a.y > b.y
      ? p.y + t > b.y && p.y - t < a.y
      : p.y + t > a.y && p.y - t < b.y;
}


bool pointOnLine2D (cv::Vec2i p, cv::Vec4i l, float t = 1E-03f)
{
  // ensure points are collinear
  double zero = (l[2] - l[0]) * (p[1] - l[1]) - (p[0] - l[0]) * (l[3] - l[1]);
  if (zero > t || zero < -t) return false;

  // check if x-coordinates are not equal
  if (l[0] - l[2] > t || l[2] - l[0] > t)
    // ensure x is between l[0] & l[2] (use tolerance)
    return l[0] > l[2]
        ? p[0] + t > l[2] && p[0] - t < l[0]
        : p[0] + t > l[0] && p[0] - t < l[2];

  // ensure y is between l[1] & l[3] (use tolerance)
  return l[1] > l[3]
      ? p[1] + t > l[3] && p[1] - t < l[1]
      : p[1] + t > l[1] && p[1] - t < l[3];
}

bool pointInLineSegment(cv::Point a, cv::Point b, cv::Point c)
{
  float crossproduct = (c.y - a.y) * (b.x - a.x) - (c.x - a.x) * (b.y - a.y);
  if (fabs(crossproduct) > 0.001f)
      return false;

  /// compare versus epsilon for floating point values, or != 0 if using integers
  if (fabs(crossproduct) > 0.001f)
    return false;

  float dotproduct = (c.x - a.x) * (b.x - a.x) + (c.y - a.y)*(b.y - a.y);
  if (dotproduct < 0)
      return false;

  float squaredlengthba = (b.x - a.x)*(b.x - a.x) + (b.y - a.y)*(b.y - a.y);
  if (dotproduct > squaredlengthba)
      return false;

  return true;
}


bool compareLines(const cv::Vec4i& l1, const cv::Vec4i& l2)
{
  int l1_c_x = (l1[0] + l1[2])/2;
  int l1_c_y = (l1[1] + l1[3])/2;
  int l2_c_x = (l2[0] + l2[2])/2;
  int l2_c_y = (l2[1] + l2[3])/2;
//  return (l1_c_y < l2_c_y);

  double l1_dist = sqrt(l1_c_x*l1_c_x + l1_c_y*l1_c_y);
  double l2_dist = sqrt(l2_c_x*l2_c_x + l2_c_y*l2_c_y);
  return l1_dist < l2_dist;

//  double a1 = atan2(l1[3]-l1[1], l1[0]-l1[2]);
//  double a2 = atan2(l2[3]-l2[1], l2[0]-l2[2]);
//  return (a1 < a2);
}






using namespace cv;
using namespace std;

Vec2d linearParameters(Vec4i line)
{
    Mat a = (Mat_<double>(2, 2) <<
                line[0], 1,
                line[2], 1);
    Mat y = (Mat_<double>(2, 1) <<
                line[1],
                line[3]);
    Vec2d mc; solve(a, y, mc);
    return mc;
}

Vec4i extendedLine(Vec4i line, double d){
    // oriented left-t-right
    Vec4d _line = line[2] - line[0] < 0 ? Vec4d(line[2], line[3], line[0], line[1]) : Vec4d(line[0], line[1], line[2], line[3]);
    double m = linearParameters(_line)[0];
    // solution of pythagorean theorem and m = yd/xd
    double xd = sqrt(d * d / (m * m + 1));
    double yd = xd * m;
    return Vec4d(_line[0] - xd, _line[1] - yd , _line[2] + xd, _line[3] + yd);
}

std::vector<Point2i> boundingRectangleContour(Vec4i line, float d){
    // finds coordinates of perpendicular lines with length d in both line points
    // https://math.stackexchange.com/a/2043065/183923

    Vec2f mc = linearParameters(line);
    float m = mc[0];
    float factor = sqrtf(
        (d * d) / (1 + (1 / (m * m)))
    );

    float x3, y3, x4, y4, x5, y5, x6, y6;
    // special case(vertical perpendicular line) when -1/m -> -infinity
    if(m == 0){
        x3 = line[0]; y3 = line[1] + d;
        x4 = line[0]; y4 = line[1] - d;
        x5 = line[2]; y5 = line[3] + d;
        x6 = line[2]; y6 = line[3] - d;
    } else {
        // slope of perpendicular lines
        float m_per = - 1/m;

        // y1 = m_per * x1 + c_per
        float c_per1 = line[1] - m_per * line[0];
        float c_per2 = line[3] - m_per * line[2];

        // coordinates of perpendicular lines
        x3 = line[0] + factor; y3 = m_per * x3 + c_per1;
        x4 = line[0] - factor; y4 = m_per * x4 + c_per1;
        x5 = line[2] + factor; y5 = m_per * x5 + c_per2;
        x6 = line[2] - factor; y6 = m_per * x6 + c_per2;
    }

    return std::vector<Point2i> {
        Point2i(x3, y3),
        Point2i(x4, y4),
        Point2i(x6, y6),
        Point2i(x5, y5)
    };
}


bool extendedBoundingRectangleLineEquivalence(const Vec4i& _l1, const Vec4i& _l2, float extensionLengthFraction, float maxAngleDiff, float boundingRectangleThickness){

    Vec4i l1(_l1), l2(_l2);
    // extend lines by percentage of line width
    float len1 = sqrtf((l1[2] - l1[0])*(l1[2] - l1[0]) + (l1[3] - l1[1])*(l1[3] - l1[1]));
    float len2 = sqrtf((l2[2] - l2[0])*(l2[2] - l2[0]) + (l2[3] - l2[1])*(l2[3] - l2[1]));
    Vec4i el1 = extendedLine(l1, len1 * extensionLengthFraction);
    Vec4i el2 = extendedLine(l2, len2 * extensionLengthFraction);

    // reject the lines that have wide difference in angles
    float a1 = atan(linearParameters(el1)[0]);
    float a2 = atan(linearParameters(el2)[0]);
    if(fabs(a1 - a2) > maxAngleDiff * M_PI / 180.0){
        return false;
    }

    // calculate window around extended line
    // at least one point needs to inside extended bounding rectangle of other line,
    std::vector<Point2i> lineBoundingContour = boundingRectangleContour(el1, boundingRectangleThickness/2);
    return
        pointPolygonTest(lineBoundingContour, cv::Point(el2[0], el2[1]), false) == 1 ||
        pointPolygonTest(lineBoundingContour, cv::Point(el2[2], el2[3]), false) == 1;
}

void drawStraightLine(cv::Mat *img, cv::Point2f p1, cv::Point2f p2, cv::Scalar color)
{
  Point2f p, q;
  // Check if the line is a vertical line because vertical lines don't have slope
  if (p1.x != p2.x)
  {
    p.x = 0;
    q.x = img->cols;
    // Slope equation (y1 - y2) / (x1 - x2)
    float m = (p1.y - p2.y) / (p1.x - p2.x);
    // Line equation:  y = mx + b
    float b = p1.y - (m * p1.x);
    p.y = m * p.x + b;
    q.y = m * q.x + b;
  }
  else
  {
    p.x = q.x = p2.x;
    p.y = 0;
    q.y = img->rows;
  }

  cv::line(*img, p, q, color, 2);
}

inline double det(double a, double b, double c, double d)
{
  return a*d - b*c;
}

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

struct IntPoint
{
  cv::Point2f p;
  float cluster;
};

bool compareIntPointsX(const IntPoint& p1, const IntPoint& p2)
{
  return p1.p.x < p2.p.x;
}

bool compareIntPointsY(const IntPoint& p1, const IntPoint& p2)
{
  return p1.p.y < p2.p.y;
}

bool equalPoints(cv::Point p1, cv::Point p2, double equal_dist = 5)
{
  double dist = sqrt( (p2.x-p1.x)*(p2.x-p1.x) + (p2.y-p1.y)*(p2.y-p1.y) );
  return dist <= equal_dist;
}

class IntLine
{
public:
  IntLine() {}
  cv::Vec4i line;
  std::vector<IntPoint> int_points;
  cv::Point center_m;

  bool operator == (const IntLine &rhs) const { return this->line == rhs.line; }
};


// TODO: cv::Exception
void processImage()
{
  if (disp_image_cv_ptr == NULL)
    return;
  if(raw_image_cv_ptr == NULL)
    return;

  if (!received_new_image || !received_new_disp_image)
    return;

  cv::Mat d_img = getDispImg(disp_img);
  cv::Mat d_float_img;
  d_img.convertTo(d_float_img, CV_32F, 1.f/255);

  cv::Mat res_img;
  raw_image_cv_ptr->image(cv::Rect(75, 10, 555, 460)).copyTo(res_img);
  res_image_test = cv::Mat(res_img.size(), CV_8UC3);
  cv::cvtColor(res_img, res_image_test, CV_GRAY2RGB);

  cv::Mat magnitude, angles;
  gradient(d_float_img, magnitude, angles);

  cv::Mat d_c_img;
  cv::Laplacian(d_img, d_c_img, CV_8UC1, 3, 1, 0, cv::BORDER_DEFAULT);
  d_c_img *= 2;
  cv::threshold(d_c_img, d_c_img, 20, 25, CV_THRESH_BINARY);
  cv::imshow("disp_filt", d_c_img);
  cv::waitKey(1);

//  cv::GaussianBlur(d_c_img, d_c_img, cv::Size(11,11), d_c_img.cols*d_c_img.rows*0.5);

  /// MORPHOLOGY TRANSFORMS
//  cv::morphologyEx(d_c_img, d_c_img, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(1, 1)));
//  cv::morphologyEx(d_c_img, d_c_img, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(1, 1)));
//  cv::morphologyEx(d_c_img, d_c_img, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(1, 1)));


  /// DEGUG HOUGH
//  debug_found_hough_val(res_image_test, d_c_img);
//  debug_found_hough_p_val(res_image_test, d_c_img);


  /// FIND HOUGH LINES AS SEGMENTS
  std::vector<cv::Vec4i> lines;
  cv::HoughLinesP(d_c_img, lines, 1, CV_PI/180, 5, 25, 20);
//  std::sort(lines.begin(), lines.end(), compareLines);
//  for( size_t i = 0; i < lines.size(); i++ )
//  {
//    cv::Vec4i l = lines[i];
//    line(res_image_test, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255,0,0), 4, CV_AA);
//    cv::putText(res_image_test, std::to_string(i), cv::Point(l[0], l[1]),
//        cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1, cvScalar(255,255,0), 1, CV_AA);
//  }


  /// FIND HOUGH LINES
//  std::vector<cv::Vec2f> full_lines;
//  cv::HoughLines(d_c_img, full_lines, 1, CV_PI/180, 65, 0);
//  cv::imshow("d_c_img", d_c_img);
//  cv::waitKey(1);


//  std::cout << "full_lines = " << full_lines.size() << std::endl;
//  std ::vector< cv::Vec2f >::const_iterator it = full_lines.begin ();
//  while ( it != full_lines .end())
//  {

//    float rho = (*it )[0];                 // first element is distance rho
//    float theta = (*it )[1]; // second element is angle theta

//    if ( theta < M_PI /4. || theta > 3.* M_PI/4. )
//    {     // ~vertical line

//      // point of intersection of the line with first row
//      cv ::Point pt1( rho / cos (theta), 0);
//      // point of intersection of the line with last row
//      cv ::Point pt2(( rho - res_image_test.rows * sin(theta )) / cos(theta ), res_image_test.rows);

//      // draw a while line
//      cv ::line( res_image_test, pt1 , pt2, cv::Scalar (0, 255, 255), 1);
//    }
//    else
//    {    //~horizontal line
//      // point of intersection of the line with first column
//      cv ::Point pt1( 0, rho / sin( theta));
//      // point of intersection of the line with last column
//      cv ::Point pt2( res_image_test.cols , ( rho - res_image_test.cols * cos(theta )) / sin(theta ));
//      // draw a white line
//      cv ::line( res_image_test, pt1 , pt2, cv::Scalar (0, 255, 255), 1);
//    }
//    ++it;
//  }

  /// FIND COUNTOURS
//  std::vector<std::vector<cv::Point>> contours;
//  cv::findContours(d_c_img.clone(), contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
//  std::vector<std::vector<cv::Point> >hull( contours.size() );
//  std::vector<std::vector<cv::Point> >aprox( contours.size() );
//  for( int i = 0; i < contours.size(); i++ )
//  {
//    cv::convexHull( cv::Mat(contours[i]), hull[i], false );
//    cv::approxPolyDP(cv::Mat(hull[i]), aprox[i], arcLength(cv::Mat(hull[i]), true) * 0.01, true);
//  }
//  for (int i = 0; i< contours.size(); i++)
//  {

//    drawContours( res_image_test, contours, i, cv::Scalar(0, 255, 0), 2 );
//    drawContours( res_image_test, aprox, i, cv::Scalar(0, 255, 255), 2 );
//  }


  /// DRAW RESULT ON IMAGE
//  magnitude *= 2;
  magnitude.convertTo(magnitude, CV_8UC1, 255);

  cv::Mat mag_noise;
  cv::threshold(magnitude, mag_noise, 1, 10, CV_THRESH_BINARY );
  magnitude -= mag_noise;
  magnitude *= 2;

//  drawVectors(res_image_test, avr_mag, avr_ang, vec_coords);

  /// Объединяем точки со значением градиента в кластеры по признаку направления градиента

  /// Создаем 8 кластеров с шагом в 45 градусов
  std::map<float, std::vector<cv::Point>> clusters;
  for(float i = 0; i < M_PI*2.0f; i+=M_PI_4)
    clusters[i] = std::vector<cv::Point>();

  /// Заполняем кластеры
  /// Проходим матрицу значий градиента и вносим значения направления градиента в соответствующий кластер
  for (int i = 0 ; i < magnitude.rows ; i += 5)
    for (int j = 0 ; j < magnitude.cols ; j+= 5)
    {
      if(magnitude.at<uchar>(i,j) < 10)
        continue;

      float ang = angles.at<float>(i,j);
      float early_cluster = 0.0f;
      for(auto &cluster : clusters)
      {
        if(ang <= cluster.first)
        {
          clusters[early_cluster].push_back(cv::Point(j,i));
          break;
        }
        early_cluster = cluster.first;
      }
    }

  magnitude.convertTo(magnitude, CV_32FC1, 1.0f/255.0f);
//  drawGradient(res_image_test, magnitude, angles, 10);

  size_t c = 0;
  std::vector<cv::Mat> mat_test_vec;
  std::vector<cv::Mat> mat_test_dist_vec;
  std::vector<std::vector<cv::Point>> contours_0;
//  std::map<float, std::vector<cv::Vec4f>> cluster_lines;
  std::map<float, std::vector<IntLine>> cluster_lines;

  std::map<float, cv::Scalar> cluster_colors;


  /// Генерируем цвета для кластеров
  cv::RNG rng(12345);
  for (auto &cluster : clusters)
  {
    cluster_colors[cluster.first] = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
  }

  /// Ищем линии в каждом кластере
  for (auto &cluster : clusters)
  {
    if (c % 2 == 1)
    {
      c++;
      continue;
    }

    std::cout << c << " cluster (" << cluster.first*180.0/M_PI << ") has " << cluster.second.size() << std::endl;

    Scalar color = cluster_colors[cluster.first];
    mat_test_vec.push_back(cv::Mat::zeros(magnitude.size(), CV_8UC1));
    mat_test_dist_vec.push_back(cv::Mat::zeros(magnitude.size(), CV_8UC1));

    for(auto p : cluster.second)
    {
//      mat_test_vec.back().at<uchar>(p.y, p.x, 0) = 255;
      cv::circle(mat_test_vec.back(), p, 3, cv::Scalar(128,128,128),CV_FILLED, 8,0);
      cv::circle(res_image_test, p, 3, color,CV_FILLED, 8,0);
    }

    cv::findContours(mat_test_vec.back(), contours_0, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

    // get the moments
    std::vector<cv::Moments> mu(contours_0.size());
    for( int i = 0; i<contours_0.size(); i++ )
    {
      mu[i] = moments( contours_0[i], false );
    }
    // get the centroid of figures.
    std::vector<cv::Point2f> mc(contours_0.size());
    for( int i = 0; i<contours_0.size(); i++)
    {
      mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
    }

    for (int i = 0; i< contours_0.size(); i++)
    {
      cv::Vec4f line;
      line[0] = mc[i].x;
      line[1] = mc[i].y;
      line[2] = mc[i].x + 100*cos(cluster.first-M_PI_2);
      line[3] = mc[i].y + 100*sin(cluster.first-M_PI_2);
      IntLine int_line; int_line.line = line; int_line.center_m = mc[i];
      cluster_lines[cluster.first].push_back(int_line);


      drawStraightLine(&mat_test_vec.back(), cv::Point(line[0], line[1]), cv::Point(line[2],line[3]), cv::Scalar(255,255,255));
      cv::line(mat_test_vec.back(),cv::Point(line[0], line[1]), cv::Point(line[2],line[3]),cv::Scalar(255,255,255),2);
      drawContours( mat_test_vec.back(), contours_0, i, cv::Scalar(255, 255, 255), 2 );
      circle( mat_test_vec.back(), mc[i], 4, cv::Scalar(255, 255, 255), -1, 8, 0 );
//      circle( res_image_test, mc[i], 8, cv::Scalar(255, 255, 255), -1, 8, 0 );
    }

//      distanceTransform(mat_test_vec.back(), mat_test_dist_vec.back(), DIST_L2, 3);
//      // Normalize the distance image for range = {0.0, 1.0}
//      // so we can visualize and threshold it
//      normalize(mat_test_dist_vec.back(), mat_test_dist_vec.back(), 0, 1.0, NORM_MINMAX);
//      threshold(mat_test_dist_vec.back(), mat_test_dist_vec.back(), 0.4, 1.0, THRESH_BINARY);
    c++;
  }
  std::cout << std::endl;

  int l = 0;
  for(auto &line_vec : cluster_lines)
  {
     Scalar color = cluster_colors[line_vec.first];
     std::cout << "Cluster " << line_vec.first*180.0/M_PI << " has " << line_vec.second.size() << " lines" << std::endl;

     for(auto &line : line_vec.second)
     {
//       drawStraightLine(&res_image_test, cv::Point(line.line[0], line.line[1]), cv::Point(line.line[2],line.line[3]), color);
//       circle( res_image_test, line.center_m, 8, cv::Scalar(255, 255, 255), -1, 8, 0 );
       cv::Point grad_end;
       grad_end.x = line.center_m.x + 30*cos(line_vec.first);
       grad_end.y = line.center_m.y + 30*sin(line_vec.first);
       cv::arrowedLine(res_image_test, line.center_m, grad_end, color, 2, 8, 0, 0.4);
       cv::putText(res_image_test, std::to_string(int(line_vec.first*180/M_PI)), grad_end,
           cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1, CV_AA);

       /// Ищем пересечение линий из текущего кластера с линиями из других кластеров
       for(auto &l_v : cluster_lines)
       {
         if(line_vec == l_v)
           continue;

         for(auto &l : l_v.second)
         {
           IntPoint p;
           bool ok = linesIntersection(line.line, l.line, p.p);
           if(!ok)
             continue;
           p.cluster = l_v.first;
//           circle( res_image_test, p.p, 4, cv::Scalar(0, 255, 0), -1, 8, 0 );

           line.int_points.push_back(p);
         }

         /// Сортируем точки пересечения
         if(int(line_vec.first*180.0/M_PI) == 90 || int(line_vec.first*180.0/M_PI) == 270)
           std::sort(line.int_points.begin(), line.int_points.end(), compareIntPointsX);
         if(int(line_vec.first*180.0/M_PI) == 0 || int(line_vec.first*180.0/M_PI) == 180)
           std::sort(line.int_points.begin(), line.int_points.end(), compareIntPointsY);
       }

       // check clusters point and center of mass
     }
  }

//  for(auto &line_vec : cluster_lines)
//  {
//    std::cout << "cluster " << line_vec.first << ": " << std::endl;
//    for(int i = 0; i < line_vec.second.size(); i++)
//    {
//      IntLine i_l = line_vec.second.at(i);
//      std::cout << "   line " << i << ": " << std::endl;
//      for(int j = 0; j < i_l.int_points.size(); j++)
//      {
//        IntPoint i_p = i_l.int_points.at(j);
//        std::cout << "      point " << j << ": " << i_p.p << std::endl;
//      }
//    }
//  }

  /// Ищем отрезки на линиях
  /// Начало отрезка - точка пересечения с предпологаемой левой или верхней границей окна
  /// Конец отрезка - точка пересечения с предпологаемой правой или нижней границей окна
  /// На отрезке должен лежать центр масс контура
  std::map<int, std::vector<cv::Vec4i>> window_lines;
  std::map<int, std::vector<cv::Point>> window_lines_centers;

  /// В каждом кластере
  for(auto &line_vec : cluster_lines)
  {
    /// На каждой линии
    for(auto &line : line_vec.second)
    {
      window_lines_centers[int(line_vec.first*180.0/M_PI)].push_back(line.center_m);
      /// Для каждой точки
      for(int i = 0; i < line.int_points.size(); i++)
      {
        IntPoint cur_p = line.int_points[i];
        if(int(line_vec.first*180.0/M_PI) == 0)
        {
          // Ищем первое пересечение  с линией из кластера 270
          if(int(cur_p.cluster*180.0/M_PI) == 270)
          {
            // Ищем подходящие отрезки на линии
            // начало - точка пересечения с кластером 270
            // конец - точка пересечения с кластером 90
            // на отрезке должен леждать центр масс контура прямой
            int j = i;
            while(j < line.int_points.size())
            {
              j++;
              IntPoint next_p = line.int_points[j];
              if(int(next_p.cluster*180.0/M_PI) == 90)
              {
                // если центр больше текущей точки и меньше следующей
                if(line.center_m.y > cur_p.p.y && line.center_m.y < next_p.p.y)
                {
                  window_lines[0].push_back(Vec4i(cur_p.p.x, cur_p.p.y, next_p.p.x, next_p.p.y));
//                  window_lines_centers[0].push_back(line.center_m);
                }
              }
            }
          }
        }

        if(int(line_vec.first*180.0/M_PI) == 90)
        {
          // Ищем первое пересечение  с линией из кластера 180
          if(int(cur_p.cluster*180.0/M_PI) == 180)
          {
            // Ищем подходящие отрезки на линии
            // начало - точка пересечения с кластером 270
            // конец - точка пересечения с кластером 90
            // на отрезке должен леждать центр масс контура прямой
            int j = i;
            while(j < line.int_points.size())
            {
              j++;
              IntPoint next_p = line.int_points[j];
              if(int(next_p.cluster*180.0/M_PI) == 0)
              {
                // если центр больше текущей точки и меньше следующей
                if(line.center_m.x > cur_p.p.x && line.center_m.x < next_p.p.x)
                {
                  //                  bottom_window.push_back(Vec4i(cur_p.p.x, cur_p.p.y, next_p.p.x, next_p.p.y));
                  window_lines[90].push_back(Vec4i(cur_p.p.x, cur_p.p.y, next_p.p.x, next_p.p.y));
//                  window_lines_centers[90].push_back(line.center_m);
                }
              }
            }
          }
        }

        if(int(line_vec.first*180.0/M_PI) == 180)
        {
          // Ищем первое пересечение  с линией из кластера 180
          if(int(cur_p.cluster*180.0/M_PI) == 270)
          {
            // Ищем подходящие отрезки на линии
            // начало - точка пересечения с кластером 270
            // конец - точка пересечения с кластером 90
            // на отрезке должен леждать центр масс контура прямой
            int j = i;
            while(j < line.int_points.size())
            {
              j++;
              IntPoint next_p = line.int_points[j];
              if(int(next_p.cluster*180.0/M_PI) == 90)
              {
                // если центр больше текущей точки и меньше следующей
                if(line.center_m.y > cur_p.p.y && line.center_m.y < next_p.p.y)
                {
                  window_lines[180].push_back(Vec4i(cur_p.p.x, cur_p.p.y, next_p.p.x, next_p.p.y));
//                  window_lines_centers[180].push_back(line.center_m);
                }
              }
            }
          }
        }
        if(int(line_vec.first*180.0/M_PI) == 270)
        {
          // Ищем первое пересечение  с линией из кластера 270
          if(int(cur_p.cluster*180.0/M_PI) == 180)
          {
            // Ищем подходящие отрезки на линии
            // начало - точка пересечения с кластером 270
            // конец - точка пересечения с кластером 90
            // на отрезке должен леждать центр масс контура прямой
            int j = i;
            while(j < line.int_points.size())
            {
              j++;
              IntPoint next_p = line.int_points[j];
              if(int(next_p.cluster*180.0/M_PI) == 0)
              {
                // если центр больше текущей точки и меньше следующей
                if(line.center_m.x > cur_p.p.x && line.center_m.x < next_p.p.x)
                {
                  //                  top_window.push_back(Vec4i(cur_p.p.x, cur_p.p.y, next_p.p.x, next_p.p.y));
                  window_lines[270].push_back(Vec4i(cur_p.p.x, cur_p.p.y, next_p.p.x, next_p.p.y));
//                  window_lines_centers[270].push_back(line.center_m);
                }
              }
            }
          }
        }
      }
    }
  }

  for(auto w_l_c : window_lines_centers)
  {
    for(auto center : w_l_c.second)
      circle( res_image_test, center, 8, cv::Scalar(255, 255, 255), -1, 8, 0 );
  }

  std::vector<std::vector<cv::Point>> windows;

  for(auto &line : window_lines)
  {
    for(auto segment : line.second)
    {
      cv::Scalar color = cluster_colors[static_cast<float>(line.first*M_PI/180.0)];
      circle( res_image_test, cv::Point(segment[0], segment[1]), 4, color, -1, 8, 0 );
      circle( res_image_test, cv::Point(segment[2], segment[3]), 4, color, -1, 8, 0 );
      cv::arrowedLine(res_image_test, cv::Point(segment[0], segment[1]),
                                      cv::Point(segment[2], segment[3]), color, 2, 8, 0, 0.1);


      if(int(line.first) == 0)
      {
        cv::Point start_0(segment[0], segment[1]);
        cv::Point end_0(segment[2], segment[3]);
        for(auto segment_270 : window_lines[270])
        {
          cv::Point start_270(segment_270[0], segment_270[1]);
          cv::Point end_270(segment_270[2], segment_270[3]);
          if(equalPoints(start_0, end_270, 10))
          {
            for(auto segment_90 : window_lines[90])
            {
              cv::Point start_90(segment_90[0], segment_90[1]);
              cv::Point end_90(segment_90[2], segment_90[3]);
              if(equalPoints(end_0, end_90, 10))
              {
                for(auto segment_180 : window_lines[180])
                {
                  cv::Point start_180(segment_180[0], segment_180[1]);
                  cv::Point end_180(segment_180[2], segment_180[3]);
                  if(equalPoints(start_90, end_180, 10))
                  {
                    if(equalPoints(start_180, start_270, 10))
                    {
                      std::vector<cv::Point> window;

                      window.push_back(start_180);
                      window.push_back(start_0);
                      window.push_back(end_0);
                      window.push_back(start_90);

                      bool left_ok = false;
                      bool right_ok = false;
                      bool top_ok = false;
                      bool bottom_ok = false;

                      for(auto w_l_c : window_lines_centers)
                        for(auto center : w_l_c.second)
                        {
//                          left_ok = ;
//                           = ;
//                           = pointInLineSegment(start_180, start_0, center);
//                           = pointInLineSegment(start_90, end_0, center);

                          if(pointInLineSegment(start_180, start_90, center))
                            left_ok = true;
                          if(pointInLineSegment(start_0, end_0, center))
                            right_ok = true;
                          if(pointInLineSegment(start_180, start_0, center))
                            top_ok = true;
                          if(pointInLineSegment(start_90, end_0, center))
                            bottom_ok = true;

                          cv::Vec4i right_line(start_0.x, start_0.y, end_0.x, end_0.y);
                          if(pointOnLine2D(center, right_line))
                            right_ok = true;

                          cv::Vec4i top_line(start_180.x, start_180.y, start_0.x, start_0.y);
                          if(pointOnLine2D(center, top_line))
                            top_ok = true;

                          cv::Vec4i bottom_line(start_90.x, start_90.y, end_0.x, end_0.y);
                          if(pointOnLine2D(center, bottom_line))
                            bottom_ok = true;
                        }

                      if(left_ok && right_ok && top_ok && bottom_ok)
                        windows.push_back(window);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  std::cout << "windows = " << windows.size() << std::endl<< std::endl;

  for(auto &window : windows)
  {
//    std::cout << std::endl << "=====" << std::endl;
//    std::cout << "window = " << window[0] << "; " << window[3] << std::endl;

//    cv::drawContours(res_image_test, window, 0, cv::Scalar(255,255,0,100), 3, 8);
//    cv::fillPoly( res_image_test, window, cv::Scalar(255,255,0,100));

    cv::rectangle(res_image_test, window[0], window[2], cv::Scalar(0,0,255,100), 3, 8,0);
    circle( res_image_test, window[0], 7, cv::Scalar(0,0,255,100), -1, 8, 0 );
    circle( res_image_test, window[1], 7, cv::Scalar(0,255,0,100), -1, 8, 0 );
    circle( res_image_test, window[2], 7, cv::Scalar(255,0,255,100), -1, 8, 0 );
    circle( res_image_test, window[3], 7, cv::Scalar(255,0,0,100), -1, 8, 0 );
  }


  /// SHOW RESULTS





  /// NEW TESTS


//  magnitude *= 4;
//  cv::morphologyEx(magnitude, magnitude, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(20, 20)));
//  cv::morphologyEx(magnitude, magnitude, cv::MORPH_TOPHAT, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(25, 25)));
////  cv::morphologyEx(magnitude, magnitude, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5)));
////    cv::morphologyEx(magnitude, magnitude, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(2, 2)));
////  cv::morphologyEx(magnitude, magnitude, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(10, 10)));
//  magnitude.convertTo(magnitude, CV_8UC1, 255);
//  cv::Mat test_i;
//  cv::threshold(magnitude, test_i, 1, 15, CV_THRESH_BINARY );
//  magnitude -= test_i;
//  magnitude *= 2;

//  debug_found_grey_value(magnitude);
//  cv::threshold(magnitude, magnitude, 5, 255, CV_THRESH_BINARY );

//  cv::Mat dist;
//  cv::distanceTransform(magnitude, dist, DIST_L2, DIST_MASK_PRECISE);//DistanceTransformMasks
//  cv::normalize(dist, dist, 0, 1.0, NORM_MINMAX);


//  cv::morphologyEx(magnitude, magnitude, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5)));
//    cv::morphologyEx(d_c_img, d_c_img, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(1, 1)));


//  debug_found_hough_p_val(res_image_test, magnitude);
//  std::vector<cv::Vec4i> test_lines;
//  cv::HoughLinesP(magnitude, test_lines, 1, CV_PI/180, 100, 25, 20);
//  std::sort(lines.begin(), lines.end(), compareLines);
//  for( size_t i = 0; i < test_lines.size(); i++ )
//  {
//    cv::Vec4i l = test_lines[i];
//    line(res_image_test, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,255,255), 1, CV_AA);
//    cv::putText(res_image_test, std::to_string(i), cv::Point(l[0], l[1]),
//        cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1, cvScalar(0,255,255), 1, CV_AA);
//  }

//  std::vector<std::vector<cv::Point>> contours;
//  cv::findContours(magnitude.clone(), contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
//  std::vector<std::vector<cv::Point> >hull( contours.size() );
//  std::vector<std::vector<cv::Point> >aprox( contours.size() );
//  for( int i = 0; i < contours.size(); i++ )
//  {
//    cv::convexHull( cv::Mat(contours[i]), hull[i], false );
//    cv::approxPolyDP(cv::Mat(hull[i]), aprox[i], arcLength(cv::Mat(hull[i]), true) * 0.01, true);
//  }
//  for (int i = 0; i< contours.size(); i++)
//  {

////    drawContours( res_image_test, contours, i, cv::Scalar(0, 255, 0), 2 );
////    drawContours( res_image_test, aprox, i, cv::Scalar(0, 255, 255), 2 );
//  }

//  magnitude.convertTo(magnitude, CV_32FC1, 1.0f/255.0f);

//  drawGradient(res_image_test, magnitude, angles, 10);
//  cv::imshow("magnitude", magnitude);
//  cv::waitKey(1);

}

void pubResImage(const ros::Publisher& pub)
{
  if (disp_image_cv_ptr == NULL)
    return;
  if (!received_new_disp_image)
    return;
  disp_image_cv_ptr->image = res_image_test;
  disp_image_cv_ptr->encoding = "bgr8";
  //  disp_image_cv_ptr->encoding = "mono8";
  pub.publish(disp_image_cv_ptr->toImageMsg());

  received_new_image = false;
  received_new_disp_image = false;
}

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

void disparityImageCallback(const stereo_msgs::DisparityImage::ConstPtr& msg)
{
  disp_img = *msg;
  disp_image_cv_ptr = cv_bridge::toCvCopy(msg->image, sensor_msgs::image_encodings::TYPE_32FC1);
  //  disp_image_cv_ptr = cv_bridge::toCvCopy(msg->image, sensor_msgs::image_encodings::TYPE_32FC1);
  //  disp_image_cv_ptr = cv_bridge::toCvCopy(msg->image, sensor_msgs::image_encodings::MONO8);
  received_new_disp_image = true;
}



int main(int argc, char** argv)
{
  ros::init(argc, argv, "window_detector_real");

  ros::NodeHandle nh;

  ros::Subscriber raw_image_sub = nh.subscribe("/elp/left/image_raw", 1000, rawImageCallback);
  ros::Subscriber disparity_image_sub = nh.subscribe("/elp/disparity", 1000, disparityImageCallback);
  ros::Publisher  res_image_pub = nh.advertise<sensor_msgs::Image>("/window_detector_image", 1);

  ros::Rate loop_rate(10);
  while(ros::ok())
  {
    processImage();

    pubResImage(res_image_pub);

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}



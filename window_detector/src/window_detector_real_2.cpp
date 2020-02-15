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
//  if(img.rows != magnitude.rows || img.cols != magnitude.cols ||
//     img.rows != angles.rows || img.cols != angles.cols ||
//     magnitude.rows != angles.rows || magnitude.cols != angles.cols)
//    return;

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

void kmean_clustering_2(cv::Mat &img, cv::Mat &res, int clusters_count)
{
  int origRows = img.rows;
  Mat colVec = img.reshape(1, img.rows*img.cols); // change to a Nx3 column vector
  Mat colVecD, bestLabels, centers, clustered;
  int attempts = 10;
  double eps = 0.001;
  colVec.convertTo(colVecD, CV_32FC3, 1.0/255.0); // convert to floating point
  double compactness = kmeans(colVecD, clusters_count, bestLabels,
                              TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, attempts, eps),
                              attempts, KMEANS_PP_CENTERS, centers);
  Mat labelsImg = bestLabels.reshape(1, origRows); // single channel image of labels

  int colors[clusters_count];
  for(int i=0; i<clusters_count; i++)
  {
    colors[i] = 255/(i+1);
  }

  res = cv::Mat(img.rows, img.cols, CV_32F);
  for(int i=0; i<img.cols*img.rows; i++)
  {
    res.at<float>(i/img.cols, i%img.cols) = (float)(colors[labelsImg.at<int>(0,i)]);
  }


  res.convertTo(res, CV_8U);
//  imshow("clustered", clustered);
//  cv::waitKey(1);
}

void colorReduce(cv::Mat& image, int div=64)
{
  int nl = image.rows;                    // number of lines
  int nc = image.cols * image.channels(); // number of elements per line

  for (int j = 0; j < nl; j++)
  {
    // get the address of row j
    uchar* data = image.ptr<uchar>(j);

    for (int i = 0; i < nc; i++)
    {
      // process each pixel
      data[i] = data[i] / div * div + div / 2;
//      data[i]= data[i] - data[i]%div + div/2;
    }
  }
}

bool greaterPointY(const cv::Point& p1, const cv::Point& p2)
{
  return p1.y > p2.y;
}

void show_histogram(std::string const& name, cv::Mat1b const& image, std::vector<int> &vals)
{
    // Set histogram bins count
    int bins = 256;
    int histSize[] = {bins};
    // Set ranges for histogram bins
    float lranges[] = {0, 256};
    const float* ranges[] = {lranges};
    // create matrix for histogram
    cv::Mat hist;
    int channels[] = {0};

    // create matrix for histogram visualization
    int const hist_height = 256;
    cv::Mat3b hist_image = cv::Mat3b::zeros(hist_height, bins);

    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

    double max_val=0;
    minMaxLoc(hist, 0, &max_val);

    std::vector<cv::Point> h_vals;
    for(int i = 0; i < bins; i+=1)
    {
      int val = cvRound(hist.at<float>(i)*hist_height/max_val);
      h_vals.push_back(cv::Point(i, val));
    }

    std::sort(h_vals.begin(), h_vals.end(), greaterPointY);
    int perv = 0;
    for(int i = 0; i < h_vals.size(); i++)
    {
      int cur = h_vals.at(i).y;
      std::cout << i <<  ": " << "cur = " << cur << "; cur-perv = " << cur-perv ;
      if(cur > 40 && abs(cur-perv) > 40)
      {
        vals.push_back(h_vals.at(i).x);
        std::cout << "   yes";
        perv = cur;
      }
      std::cout << std::endl;

      if(cur < 30)
        break;
    }

    // visualize each bin
    for(int b = 0; b < bins; b++)
    {
      float const binVal = hist.at<float>(b);
      int   const height = cvRound(binVal*hist_height/max_val);
      cv::line ( hist_image, cv::Point(b, hist_height-height),
                 cv::Point(b, hist_height), cv::Scalar::all(255));
    }
    cv::imshow(name, hist_image);
    cv::waitKey(1);
}

double norm_ang(double ang)
{
  while(ang < 0.0)
    ang += M_PI*2;
  while(ang > M_PI*2)
    ang -= M_PI*2;
  return ang;
}

bool pointInPolygon(cv::Point p, std::vector<cv::Point> &polygon)
{
  bool in_range = false;
  if(polygon.size() < 3)
    return in_range;
  for(size_t i = 1; i < polygon.size(); i++)
  {
    if( ( ((p.y >= polygon.at(i).y) && (p.y < polygon.at(i-1).y))  || ((p.y >= polygon.at(i-1).y) && (p.y < polygon.at(i).y)) ) &&
        ( p.x > (polygon[i-1].x - polygon[i].x) * (p.y - polygon[i].y) / (polygon[i - 1].y - polygon[i].y) + polygon[i].x ) )
      in_range = !in_range;
  }

  return in_range;
}

// TODO: cv::Exception
void processImage()
{
  /// Проверки наличия изображения
  if (disp_image_cv_ptr == NULL)
    return;
  if(raw_image_cv_ptr == NULL)
    return;
  if (!received_new_image || !received_new_disp_image)
    return;

  /// Преобразование disparity map в изображение
  cv::Mat d_img = getDispImg(disp_img);

  cv::GaussianBlur(d_img, d_img, cv::Size(21,21), d_img.cols*d_img.rows*0.5);

  /// Морфологические преобразования
  // cv::morphologyEx(d_img, d_img, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(60, 60)));
  // cv::morphologyEx(d_img, d_img, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(40, 40)));
  // cv::morphologyEx(d_img, d_img, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(20, 20)));

  /// Изображения для вывода результата
  cv::Mat res_img;
  raw_image_cv_ptr->image(cv::Rect(75, 10, 555, 460)).copyTo(res_img);
  res_image_test = cv::Mat(res_img.size(), CV_8UC3);
  cv::cvtColor(res_img, res_image_test, CV_GRAY2RGB);

  cv::Mat disp_3_ch = cv::Mat(d_img.size(), CV_8UC3);
  cv::cvtColor(d_img, disp_3_ch, CV_GRAY2RGB);

  cv::imshow("d_img", d_img);
  cv::waitKey(1);

  /// Поиск градиента
  cv::Mat d_float_img;
  d_img.convertTo(d_float_img, CV_32F, 1.f/255);
  cv::Mat magnitude, angles;
  gradient(d_float_img, magnitude, angles);

  cv::Mat ch_mag;
  magnitude.convertTo(ch_mag, CV_8UC1, 255);
  ch_mag *= 2;
  cv::threshold(ch_mag, ch_mag, 15, 255, CV_THRESH_BINARY);
  std::vector<std::vector<cv::Point>> mag_cnts;
  cv::findContours(ch_mag, mag_cnts, CV_RETR_LIST , CV_CHAIN_APPROX_TC89_L1);

  cv::imshow("magnitude", ch_mag);
  cv::waitKey(1);

  /// Построение гистограммы
//  std::vector<int> hist_vals;
//  show_histogram("histogramm", d_img, hist_vals);
//  std::cout << "--- hist_vals = " << hist_vals.size() << std::endl;

//  cv::Mat test_small, test_origin, clustered;
//  cv::resize(d_img, test_small, cv::Size(d_img.cols * 0.25, d_img.rows * 0.25), 0, 0, CV_INTER_LINEAR);
//  int cluster_cout = hist_vals.size() > 1 ? hist_vals.size() : 2;
//  kmean_clustering_2(test_small, test_origin, cluster_cout);
//  cv::resize(test_origin, clustered, cv::Size(test_origin.cols * 4, test_origin.rows * 4), 0, 0, CV_INTER_LINEAR);

//  cv::Mat clustering_3_ch = cv::Mat(clustered.size(), CV_8UC3);
//  cv::cvtColor(clustered, clustering_3_ch, CV_GRAY2RGB);
//  //drawGradient(clustering_3_ch, magnitude, angles);
//  cv::imshow("clustering_3_ch", clustering_3_ch);
//  cv::waitKey(1);


  cv::Mat test_reduce;
  d_img.copyTo(test_reduce);
  colorReduce(test_reduce, 8);
  cv::imshow("test_reduce", test_reduce);
  cv::waitKey(1);

  /// Поиск градиента
  cv::Mat reduce_float_img;
  test_reduce.convertTo(reduce_float_img, CV_32F, 1.f/255);
  cv::Mat magnitude_2, angles_2;
  gradient(reduce_float_img, magnitude_2, angles_2);


  cv::Mat test_grad_1;
  res_image_test.copyTo(test_grad_1);
  drawGradient(test_grad_1, magnitude_2, angles_2);
  cv::imshow("GRAD REDUCE", test_grad_1);

  cv::Mat test_grad_2;
  res_image_test.copyTo(test_grad_2);
  drawGradient(test_grad_2, magnitude, angles);
  cv::imshow("ORIGIN REDUCE", test_grad_2);





//  /// Сегментация изображения
//  std::map<int, std::vector<cv::Point>> segments;
//  std::map<int, cv::Scalar> segments_colors;
//  cv::RNG rng(12345);
//  for(int i = 0; i < hist_vals.size(); i++)
//  {
//    segments[hist_vals.at(i)] = std::vector<cv::Point>();
//    segments_colors[hist_vals.at(i)] = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
//  }

//  for(int y = 0; y < d_img.rows; y+=5)
//    for(int x = 0; x < d_img.cols; x+=5)
//    {
//      //int early_s = 0;
//      for(auto &s : segments)
//      {
//        //if(d_img.at<uchar>(y,x) <= s.first)
//        if(abs(d_img.at<uchar>(y,x) - s.first) <= 5)
//        {
//          segments[s.first].push_back(cv::Point(x,y));
//          break;
//        }
//        //early_s = s.first;
//      }
//    }

//  //drawGradient(res_image_test, magnitude, angles);

//  /// Анализ сегментов изображения
//  /// Разделение сегментов в зависимости от направления градиента на границах сегментов
//  double sum = 0;
//  int segment_num = 0;
//  bool is_obstacle = false;
//  std::map<int, std::vector<cv::Mat>> segments_imgs;
//  for(auto s : segments)
//  {
//    //    if(segment_num != 2)
//    //    {
//    //      segment_num++;
//    //      continue;
//    //    }
//    segment_num++;

//    Scalar color = segments_colors[s.first];
//    segments_imgs[s.first].push_back(cv::Mat::zeros(d_img.size(), d_img.type()));
//    for(auto p : s.second)
//    {
//      cv::circle(disp_3_ch, p, 3, color, CV_FILLED, 8,0);
//      cv::circle(segments_imgs[s.first].back(), p, 3, cv::Scalar(255,255,255), CV_FILLED, 8,0);
//    }

//    std::vector<std::vector<cv::Point>> cnts;
//    cv::findContours(segments_imgs[s.first].back(), cnts, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
//    // std::vector<std::vector<cv::Point>> hull(cnts.size());
//    std::vector<std::vector<cv::Point>> aprox(cnts.size());
//    std::cout << s.first  << ":" << std::endl;
//    for(size_t i = 0; i < cnts.size(); i++)
//    {
//      //drawContours( res_image_test, cnts, i, color, 2 );

//      std::cout << "   " << i  << ":" << std::endl;
//      cv::approxPolyDP(cv::Mat(cnts[i]), aprox[i], 5, true);
//      cv::Moments mu_i = cv::moments(aprox[i], false);
//      Point2f mc_i =  Point2f( mu_i.m10/mu_i.m00 , mu_i.m01/mu_i.m00 );
//      cv::Point early_cnt(aprox.at(i).at(0));

//      for(size_t j = 1; j < aprox.at(i).size(); j++)
//      {
//        cv::Point cnt = aprox.at(i).at(j);

//        float cnt_ang = atan2(cnt.y-early_cnt.y, cnt.x-early_cnt.x);
//        float grad_ang = angles.at<float>(early_cnt.y,early_cnt.x);

//        double ang_deg = norm_ang(cnt_ang-grad_ang)*180/M_PI;
//        std::cout << "      " << j  << ":" << ang_deg << std::endl;
//        sum += (cnt_ang-grad_ang);



//        //if(fabs(ang_deg-0) > 0.0000001 && fabs(ang_deg-180) > 0.0000001 &&
//        //   fabs(ang_deg-90) > 0.0000001 && fabs(ang_deg-270) > 0.0000001)
//        //{
//        //  if(ang_deg  >= 0 && ang_deg  <= 180)
//        //    is_obstacle = true;
//        //}



//        /// Рисуем градиент для каждого сегмента контура
//        //cv::Point2f p2(cnt.x + 300*magnitude.at<float>(cnt.y,cnt.x)*cos(angles.at<float>(cnt.y,cnt.x)),
//        //               cnt.y + 300*magnitude.at<float>(cnt.y,cnt.x)*sin(angles.at<float>(cnt.y,cnt.x)));
//        cv::Point2f p2(cnt.x + 10*cos(angles.at<float>(cnt.y,cnt.x)),
//                       cnt.y + 10*sin(angles.at<float>(cnt.y,cnt.x)));

//        if (pointInPolygon(p2, aprox.at(i)))
//        {
//          is_obstacle = true;
//          cv::circle(disp_3_ch, p2, 4, cv::Scalar(0,0,0), CV_FILLED, 8,0);
//          cv::circle(disp_3_ch, p2, 2, color, CV_FILLED, 8,0);
//          cv::circle(res_image_test, p2, 3, cv::Scalar(255,255,0), CV_FILLED, 8,0);
//        }

//        //cv::arrowedLine(res_image_test, cnt, p2, color, 2, 8, 0, 0.4);
//        cv::arrowedLine(disp_3_ch, cnt, p2, cv::Scalar(255,0,0), 1, 8, 0, 0.4);
//        cv::arrowedLine(res_image_test, early_cnt, cnt, color, 1, 8, 0, 0.4);
//        //cv::arrowedLine(disp_3_ch, early_cnt, cnt, cv::Scalar(0,255,0), 2, 8, 0, 0.4);
//        //std::string text_ang = std::to_string(j) + "_" + std::to_string(int((cnt_ang-grad_ang)*180/M_PI));
//        //cv::putText(disp_3_ch, text_ang, p2, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0,255,0), 1, CV_AA);


//        if(is_obstacle)
//          drawContours( res_image_test, aprox, i, cv::Scalar(0,0,255,100), 2);
//        else
//          drawContours( res_image_test, aprox, i, cv::Scalar(0,255,0,100), CV_FILLED);
//        //        cv::arrowedLine(res_image_test, cnt, p2, color, 1, 8, 0, 0.4);
//        //        is_obstacle = false;
//        early_cnt = cnt;

//      }
//      //      if(is_obstacle)
//      //        drawContours( res_image_test, cnts, i, cv::Scalar(0,0,255,100), CV_FILLED);
//      //      else
//      //        drawContours( res_image_test, cnts, i, cv::Scalar(0,255,0,100), CV_FILLED);
//      is_obstacle = false;
//      std::cout << "      SUM = " << sum << std::endl;

//      //cv::convexHull( cv::Mat(cnts[i]), hull[i], false );
//      //cv::approxPolyDP(cv::Mat(hull[i]), aprox[i], arcLength(cv::Mat(hull[i]), true) * 0.01, true);
//      //cv::approxPolyDP(cv::Mat(cnts[i]), aprox[i], 5, true);

//      drawContours( disp_3_ch, aprox, i, cv::Scalar(0,0,255), 1);
//      cv::circle(disp_3_ch, mc_i, 3, cv::Scalar(0,0,255), CV_FILLED, 8,0);
//      //std::string text_c = std::to_string(s.first) + "_" + std::to_string(i);
//      std::string text_c = std::to_string(segment_num) + "_" + std::to_string(s.first);
//      cv::putText(disp_3_ch, text_c, mc_i, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255,0,0), 1, CV_AA);

//      sum = 0;
//    }
//  }

//  //drawGradient(disp_3_ch, magnitude, angles);

//  std::cout << "=================================================" << std::endl << std::endl << std::endl << std::endl;
//  cv::imshow("kneam 0", disp_3_ch);
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
  received_new_disp_image = true;
}



int main(int argc, char** argv)
{
  ros::init(argc, argv, "window_detector_real_2");

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



//using namespace cv;
//int main( int argc, char** argv )
//{
//  Mat src = imread( "/home/gek/drone_cup/thirdparty/img/7.png" );
//  imshow( "src image", src );
//  waitKey( 0 );
//  cv::Mat new_image;
//  kmean_clustering(src, new_image, 4);
//  imshow( "clustered image", new_image );
//  waitKey( 0 );
////  Mat samples(src.rows * src.cols, 3, CV_32F);
////  for( int y = 0; y < src.rows; y++ )
////    for( int x = 0; x < src.cols; x++ )
////      for( int z = 0; z < 3; z++)
////        samples.at<float>(y + x*src.rows, z) = src.at<Vec3b>(y,x)[z];


////  int clusterCount = 2;
////  Mat labels;
////  int attempts = 5;
////  Mat centers;
////  cv::kmeans(samples, clusterCount, labels,
////         TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers );

////  Mat new_image( src.size(), src.type() );
////  for( int y = 0; y < src.rows; y++ )
////    for( int x = 0; x < src.cols; x++ )
////    {
////      int cluster_idx = labels.at<int>(y + x*src.rows,0);
////      new_image.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
////      new_image.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
////      new_image.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
////    }
////  imshow( "clustered image", new_image );
////  waitKey( 0 );
//}

//int main(int argc, char** argv)
//{
//    Mat src = imread( "/home/gek/drone_cup/thirdparty/img/1.png" );
//    if( src.empty() )
//    {
//        return -1;
//    }
//    vector<Mat> bgr_planes;
//    split( src, bgr_planes );
//    int histSize = 256;
//    float range[] = { 0, 256 }; //the upper boundary is exclusive
//    const float* histRange = { range };
//    bool uniform = true, accumulate = false;
//    Mat b_hist, g_hist, r_hist;
//    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
//    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
//    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
//    int hist_w = 512, hist_h = 400;
//    int bin_w = cvRound( (double) hist_w/histSize );
//    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
//    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
//    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
//    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
//    for( int i = 1; i < histSize; i++ )
//    {
//        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ),
//              Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
//              Scalar( 255, 0, 0), 2, 8, 0  );
//        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ),
//              Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
//              Scalar( 0, 255, 0), 2, 8, 0  );
//        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ),
//              Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
//              Scalar( 0, 0, 255), 2, 8, 0  );
//    }
//    imshow("Source image", src );
//    imshow("calcHist Demo", histImage );
//    waitKey();
//    return 0;
//}

//#include <cv.h>
//#include <highgui.h>
//#include <stdlib.h>
//#include <stdio.h>

//#include <vector>
//#include <algorithm>

//#define RECT_COLORS_SIZE 10

//// получение пикселя изображения (по типу картинки и координатам)
//#define CV_PIXEL(type,img,x,y) (((type*)((img)->imageData+(y)*(img)->widthStep))+(x)*(img)->nChannels)

//// Various color types
////          0                   1         2              3              4                5                6             7               8               9                       10
//enum {cBLACK=0, cWHITE, cGREY, cRED, cORANGE, cYELLOW, cGREEN, cAQUA, cBLUE, cPURPLE, NUM_COLOR_TYPES};
//char* sCTypes[NUM_COLOR_TYPES] = {"Black", "White","Grey","Red","Orange","Yellow","Green","Aqua","Blue","Purple"};
//uchar cCTHue[NUM_COLOR_TYPES] =    {0,       0,      0,     0,     20,      30,      60,    85,   120,    138  };
//uchar cCTSat[NUM_COLOR_TYPES] =    {0,       0,      0,    255,   255,     255,     255,   255,   255,    255  };
//uchar cCTVal[NUM_COLOR_TYPES] =    {0,      255,    120,   255,   255,     255,     255,   255,   255,    255  };

//typedef unsigned int uint;

//// число пикселей данного цвета на изображении
//uint colorCount[NUM_COLOR_TYPES] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

//// Determine what type of color the HSV pixel is. Returns the colorType between 0 and NUM_COLOR_TYPES.
//int getPixelColorType(int H, int S, int V)
//{
//  int color = cBLACK;

//#if 1
//  if (V < 75)
//    color = cBLACK;
//  else if (V > 190 && S < 27)
//    color = cWHITE;
//  else if (S < 53 && V < 185)
//    color = cGREY;
//  else
//#endif
//  {
//    if (H < 7)
//      color = cRED;
//    else if (H < 25)
//      color = cORANGE;
//    else if (H < 34)
//      color = cYELLOW;
//    else if (H < 73)
//      color = cGREEN;
//    else if (H < 102)
//      color = cAQUA;
//    else if (H < 140)
//      color = cBLUE;
//    else if (H < 170)
//      color = cPURPLE;
//    else    // full circle
//      color = cRED;   // back to Red
//  }
//  return color;
//}

//// сортировка цветов по количеству
//bool colors_sort(std::pair< int, uint > a, std::pair< int, uint > b)
//{
//  return (a.second > b.second);
//}

//int main(int argc, char* argv[])
//{
//  // для хранения изображения
//  IplImage* image=0, *hsv=0, *dst=0, *dst2=0, *color_indexes=0, *dst3=0;

//  //
//  // загрузка изображения
//  //

//  char img_name[] =  "/home/gek/drone_cup/thirdparty/img/5.png";

//  // имя картинки задаётся первым параметром
//  char* image_filename = argc >= 2 ? argv[1] : img_name;

//  // получаем картинку
//  image = cvLoadImage(image_filename, 1);

//  printf("[i] image: %s\n", image_filename);
//  if(!image){
//    printf("[!] Error: cant load test image: %s\n", image_filename);
//    return -1;
//  }

//  // показываем картинку
//  cvNamedWindow("image");
//  cvShowImage("image", image);

//  //
//  // преобразуем изображение в HSV
//  //
//  hsv = cvCreateImage( cvGetSize(image), IPL_DEPTH_8U, 3 );
//  cvCvtColor( image, hsv, CV_BGR2HSV );

//  // картинки для хранения результатов
//  dst = cvCreateImage( cvGetSize(image), IPL_DEPTH_8U, 3 );
//  dst2 = cvCreateImage( cvGetSize(image), IPL_DEPTH_8U, 3 );
//  color_indexes = cvCreateImage( cvGetSize(image), IPL_DEPTH_8U, 1 ); //для хранения индексов цвета

//  // для хранения RGB-х цветов
//  CvScalar rgb_colors[NUM_COLOR_TYPES];

//  int i=0, j=0, x=0, y=0;

//  // обнуляем цвета
//  for(i=0; i<NUM_COLOR_TYPES; i++) {
//    rgb_colors[i] = cvScalarAll(0);
//  }

//  for (y=0; y<hsv->height; y++) {
//    for (x=0; x<hsv->width; x++) {

//      // получаем HSV-компоненты пикселя
//      uchar H = CV_PIXEL(uchar, hsv, x, y)[0];        // Hue
//      uchar S = CV_PIXEL(uchar, hsv, x, y)[1];        // Saturation
//      uchar V = CV_PIXEL(uchar, hsv, x, y)[2];        // Value (Brightness)

//      // определяем к какому цвету можно отнести данные значения
//      int ctype = getPixelColorType(H, S, V);

//      // устанавливаем этот цвет у отладочной картинки
//      CV_PIXEL(uchar, dst, x, y)[0] = cCTHue[ctype];  // Hue
//      CV_PIXEL(uchar, dst, x, y)[1] = cCTSat[ctype];  // Saturation
//      CV_PIXEL(uchar, dst, x, y)[2] = cCTVal[ctype];  // Value

//      // собираем RGB-составляющие
//      rgb_colors[ctype].val[0] += CV_PIXEL(uchar, image, x, y)[0]; // B
//      rgb_colors[ctype].val[1] += CV_PIXEL(uchar, image, x, y)[1]; // G
//      rgb_colors[ctype].val[2] += CV_PIXEL(uchar, image, x, y)[2]; // R

//      // сохраняем к какому типу относится цвет
//      CV_PIXEL(uchar, color_indexes, x, y)[0] = ctype;

//      // подсчитываем :)
//      colorCount[ctype]++;
//    }
//  }

//  // усреднение RGB-составляющих
//  for(i=0; i<NUM_COLOR_TYPES; i++) {
//    rgb_colors[i].val[0] /= colorCount[i];
//    rgb_colors[i].val[1] /= colorCount[i];
//    rgb_colors[i].val[2] /= colorCount[i];
//  }

//  // теперь загоним массив в вектор и отсортируем :)
//  std::vector< std::pair< int, uint > > colors;
//  colors.reserve(NUM_COLOR_TYPES);

//  for(i=0; i<NUM_COLOR_TYPES; i++){
//    std::pair< int, uint > color;
//    color.first = i;
//    color.second = colorCount[i];
//    colors.push_back( color );
//  }
//  // сортируем
//  std::sort( colors.begin(), colors.end(), colors_sort );

//  // для отладки - выводим коды, названия цветов и их количество
//  for(i=0; i<colors.size(); i++){
//    printf("[i] color %d (%s) - %d\n", colors[i].first, sCTypes[colors[i].first], colors[i].second );
//  }

//  // выдаём код первых цветов
//  printf("[i] color code: \n");
//  for(i=0; i<NUM_COLOR_TYPES; i++)
//    printf("%02d ", colors[i].first);
//  printf("\n");
//  printf("[i] color names: \n");
//  for(i=0; i<NUM_COLOR_TYPES; i++)
//    printf("%s ", sCTypes[colors[i].first]);
//  printf("\n");

//  // покажем цвета
//  cvZero(dst2);
//  int h = dst2->height / RECT_COLORS_SIZE;
//  int w = dst2->width;
//  for(i=0; i<RECT_COLORS_SIZE; i++ ){
//    cvRectangle(dst2, cvPoint(0, i*h), cvPoint(w, i*h+h), rgb_colors[colors[i].first], -1);
//  }
//  cvShowImage("colors", dst2);
//  //cvSaveImage("dominate_colors_table.png", dst2);

//  // покажем картинку в найденных цветах
//  dst3 = cvCloneImage(image);
//  for (y=0; y<dst3->height; y++) {
//    for (x=0; x<dst3->width; x++) {
//      int color_index = CV_PIXEL(uchar, color_indexes, x, y)[0];

//      CV_PIXEL(uchar, dst3, x, y)[0] = rgb_colors[color_index].val[0];
//      CV_PIXEL(uchar, dst3, x, y)[1] = rgb_colors[color_index].val[1];
//      CV_PIXEL(uchar, dst3, x, y)[2] = rgb_colors[color_index].val[2];
//    }
//  }

//  cvNamedWindow("dst3");
//  cvShowImage("dst3", dst3);
//  //cvSaveImage("dominate_colors.png", dst3);

//  // конвертируем отладочную картинку обратно в RGB
//  cvCvtColor( dst, dst, CV_HSV2BGR );

//  // показываем результат
//  cvNamedWindow("color");
//  cvShowImage("color", dst);

//  // ждём нажатия клавиши
//  cvWaitKey(0);

//  // освобождаем ресурсы
//  cvReleaseImage(&image);
//  cvReleaseImage(&hsv);
//  cvReleaseImage(&dst);
//  cvReleaseImage(&dst2);
//  cvReleaseImage(&color_indexes);
//  cvReleaseImage(&dst3);

//  // удаляем окна
//  cvDestroyAllWindows();
//  return 0;
//}

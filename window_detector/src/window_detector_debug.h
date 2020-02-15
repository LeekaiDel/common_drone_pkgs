#ifndef WINDOW_DETECTOR_DEBUG_H__
#define WINDOW_DETECTOR_DEBUG_H__

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>

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

/*
 * Draw gradient on img as red arrows
 */
void drawGradient(const cv::Mat &img, cv::Mat& magnitude, cv::Mat& angles, int flow_resolution=10)
{
  cv::Mat d_img = cv::Mat(img.size(), CV_8UC3);
  cv::cvtColor(img, d_img, CV_GRAY2RGB);

  if(d_img.rows != magnitude.rows || d_img.cols != magnitude.cols ||
     d_img.rows != angles.rows || d_img.cols != angles.cols ||
     magnitude.rows != angles.rows || magnitude.cols != angles.cols)
    return;

  for (int i = 0 ; i < d_img.rows ; i += flow_resolution)
    for (int j = 0 ; j < d_img.cols ; j+= flow_resolution)
    {
      if(magnitude.at<float>(i,j) < 0.001)
        continue;
      cv::Point2f p(j,i);
      cv::Point2f p2(p.x + 30*magnitude.at<float>(i,j)*cos(angles.at<float>(i,j)),
                     p.y + 30*magnitude.at<float>(i,j)*sin(angles.at<float>(i,j)));

      cv::arrowedLine(d_img, p, p2, cv::Scalar(0,0,255), 1, 8, 0, 0.4);
    }

  cv::imshow("Gradient", d_img);
  cv::waitKey(1);
}

/*
 * Draw contours in image with help random colors
 */
void drawMyContours(const cv::Mat &img, const std::vector<std::vector<cv::Point>> &cnts)
{
  cv::Mat test_cnt;
  img.copyTo(test_cnt);
  cv::RNG rng_c(12345);
  for(size_t i = 0; i < cnts.size(); i++)
  {
    cv::Scalar color = cv::Scalar(rng_c.uniform(0,255), rng_c.uniform(0, 255), rng_c.uniform(0, 255));
    drawContours(test_cnt, cnts, static_cast<int>(i), color, 2);
  }
  cv::imshow("Contours", test_cnt);
  cv::waitKey(1);
}

#endif // WINDOW_DETECTOR_DEBUG_H__

#ifndef DELAUNAY2D_
#define DELAUNAY2D_
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

void draw_subdiv_point( cv::Mat& img, cv::Point2f fp, cv::Scalar color );
void draw_subdiv( cv::Mat& img, cv::Subdiv2D& subdiv, cv::Scalar delaunay_color );
void locate_point( cv::Mat& img, cv::Subdiv2D& subdiv, cv::Point2f fp, cv::Scalar active_color );
void paint_voronoi( cv::Mat& img, cv::Subdiv2D& subdiv );
void build_tri_face_data(cv::Subdiv2D& subdiv, std::vector<cv::Point2f> &pol, std::vector<cv::Vec3i> &faces, cv::Size &imsize);
void draw_subdiv(cv::Mat &img, std::vector<cv::Point2f> &pol, std::vector<cv::Vec3i> &faces, cv::Scalar delaunay_color);

template <typename T>
int build_delaunay(cv::Mat &img, std::vector<cv::Point_ <T> > &pol, cv::Subdiv2D &subdiv)
{
    //help();

	bool debug_flag = false;

    cv::Scalar active_facet_color(0, 0, 255), delaunay_color(255,255,255);
	int rows = img.rows;
	int cols = img.cols;
    cv::Rect rect(0, 0, cols, rows);

    subdiv=cv::Subdiv2D(rect);
    std::string win = "Delaunay Demo";

	int N = pol.size();
    for( int i = 0; i < N; i++ )
    {
        cv::Point2f fp((float)pol[i].x,(float)pol[i].y);
		if(fp.x>cols-1){fp.x=cols-1;}
		if(fp.y>rows-1){fp.y=rows-1;}
        subdiv.insert(fp);
    }

	if(debug_flag == true){
		cv::Mat drawing = img.clone();
		draw_subdiv( drawing, subdiv, delaunay_color );
		cv::imshow( win, drawing );
		cv::waitKey(0);
	}

    return 0;
}

#endif
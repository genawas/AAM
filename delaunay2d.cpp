#include "delaunay2d.h"

static void help()
{
	std::cout << "\nThis program demostrates iterative construction of\n"
		"delaunay triangulation and voronoi tesselation.\n"
		"It draws a random set of points in an image and then delaunay triangulates them.\n"
		"Usage: \n"
		"./delaunay \n"
		"\nThis program builds the traingulation interactively, you may stop this process by\n"
		"hitting any key.\n";
}

void draw_subdiv_point( cv::Mat& img, cv::Point2f fp, cv::Scalar color )
{
	circle( img, fp, 3, color, CV_FILLED, 8, 0 );
}

void draw_subdiv(cv::Mat &img, cv::Subdiv2D& subdiv, cv::Scalar delaunay_color)
{
	bool draw;
	std::vector<cv::Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	std::vector<cv::Point> pt(3);

	for(size_t i = 0; i < triangleList.size(); ++i)
	{
		cv::Vec6f t = triangleList[i];

		pt[0] = cv::Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = cv::Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = cv::Point(cvRound(t[4]), cvRound(t[5]));

		draw=true;

		for(int i=0;i<3;i++){
			if(pt[i].x>img.cols||pt[i].y>img.rows||pt[i].x<0||pt[i].y<0)
				draw=false;
		}
		if (draw){
			line(img, pt[0], pt[1], delaunay_color, 1);
			line(img, pt[1], pt[2], delaunay_color, 1);
			line(img, pt[2], pt[0], delaunay_color, 1);
		}
	}
}

void draw_subdiv(cv::Mat &img, std::vector<cv::Point2f> &pol, std::vector<cv::Vec3i> &faces, cv::Scalar delaunay_color)
{
	bool draw;

	std::vector<cv::Point> pt(3);

	for(size_t i = 0; i < faces.size(); ++i)
	{
		cv::Vec3i f = faces[i];
		
		draw=true;
		for(int j=0;j<3;j++){
			int ind = f[j];
			pt[j] = cv::Point(cvRound(pol[ind].x), cvRound(pol[ind].y));
			
			if(pt[j].x>img.cols||pt[j].y>img.rows||pt[j].x<0||pt[j].y<0)
				draw=false;
		}

		if (draw){
			line(img, pt[0], pt[1], delaunay_color, 1);
			line(img, pt[1], pt[2], delaunay_color, 1);
			line(img, pt[2], pt[0], delaunay_color, 1);
		}
	}
}

int return_nearest(cv::Point2f &pt, std::vector<cv::Point2f> &pol)
{
	float eps = 0.0001f;

	for(int i=0;i<pol.size();i++){
		float x = pt.x - pol[i].x;
		float y = pt.y - pol[i].y;
		float d = sqrt(x*x+y*y);
		if(d<eps){
			return i;
		}
	}
	return -1;
}

void build_tri_face_data(cv::Subdiv2D& subdiv, std::vector<cv::Point2f> &pol, std::vector<cv::Vec3i> &faces, cv::Size &imsize)
{

	int e0=0, vertex=0;
	cv::Point2f fp = pol[0];
	std::vector<cv::Vec6f> tList;
	subdiv.getTriangleList(tList);

	for(size_t i = 0; i < tList.size(); ++i)
	{
		cv::Vec6f t = tList[i];
		cv::Point2f pt;
		cv::Vec3i F;
		bool flag = true;
		for(int j=0;j<3;j++){
			
			pt = cv::Point2f(t[2*j], t[2*j+1]);
			if(pt.x>imsize.width||pt.y>imsize.height||pt.x<0||pt.y<0){
				flag = false;
				break;
			}
			
			int ind = return_nearest(pt, pol);
			assert(ind>-1);
			F[j] = ind;
		}
		if(flag == true){
			faces.push_back(F);
		}
	}
}

void locate_point( cv::Mat& img, cv::Subdiv2D& subdiv, cv::Point2f fp, cv::Scalar active_color )
{
	int e0=0, vertex=0;

	subdiv.locate(fp, e0, vertex);

	if( e0 > 0 )
	{
		int e = e0;
		do
		{
			cv::Point2f org, dst;
			if( subdiv.edgeOrg(e, &org) > 0 && subdiv.edgeDst(e, &dst) > 0 )
				line( img, org, dst, active_color, 3, CV_AA, 0 );

			e = subdiv.getEdge(e, cv::Subdiv2D::NEXT_AROUND_LEFT);
		}
		while( e != e0 );
	}

	draw_subdiv_point( img, fp, active_color );
}

void paint_voronoi( cv::Mat& img, cv::Subdiv2D& subdiv )
{
	std::vector<std::vector<cv::Point2f> > facets;
	std::vector<cv::Point2f> centers;
	subdiv.getVoronoiFacetList(std::vector<int>(), facets, centers);

	std::vector<cv::Point> ifacet;
	std::vector<std::vector<cv::Point> > ifacets(1);

	for( size_t i = 0; i < facets.size(); i++ )
	{
		ifacet.resize(facets[i].size());
		for( size_t j = 0; j < facets[i].size(); j++ )
			ifacet[j] = facets[i][j];

		cv::Scalar color;
		color[0] = rand() & 255;
		color[1] = rand() & 255;
		color[2] = rand() & 255;
		cv::fillConvexPoly(img, ifacet, color, 8, 0);

		ifacets[0] = ifacet;
		polylines(img, ifacets, true, cv::Scalar(), 1, CV_AA, 0);
		circle(img, centers[i], 3, cv::Scalar(), -1, CV_AA, 0);
	}
}

void warpTextureFromTriangle(std::vector<cv::Point2f> &srcTri, cv::Mat &originalImage,
							 std::vector<cv::Point2f> &dstTri, cv::Mat &warp_final)
{
	cv::Mat warp_mat(2, 3, CV_32FC1);
	cv::Mat warp_dst, warp_mask;
	CvPoint trianglePoints[3];
	trianglePoints[0] = dstTri[0];
	trianglePoints[1] = dstTri[1];
	trianglePoints[2] = dstTri[2];
	warp_dst = cv::Mat::zeros(warp_final.rows, warp_final.cols,
		originalImage.type());
	warp_mask = cv::Mat::zeros(warp_final.rows, warp_final.cols,
		originalImage.type());
	/// Get the Affine Transform
	warp_mat = getAffineTransform(&srcTri[0], &dstTri[0]);
	/// Apply the Affine Transform to the src image
	warpAffine(originalImage, warp_dst, warp_mat, warp_dst.size());
	cvFillConvexPoly(new IplImage(warp_mask), trianglePoints, 3,
		CV_RGB(255,255,255), CV_AA, 0);
	//warp_mask.convertTo(warp_mask, CV_64FC1);
	//imshow("mask",warp_mask);cv::waitKey();
	warp_dst.copyTo(warp_final, warp_mask);
}

void wrap_piecewise_nonlin(cv::Mat &img_in, cv::Mat &img_out, std::vector<cv::Vec3i> &F, 
						   std::vector<cv::Point2f> &cp_src, std::vector<cv::Point2f> &cp_dst)
{
	std::vector<cv::Point2f> pt_src(3);
	std::vector<cv::Point2f> pt_dst(3);

	//img_out = cv::Mat::zeros(img_in.rows, img_in.cols, img_in.type());
	//cv::Mat temp = img_in.clone();
	for(size_t i = 0; i < F.size(); ++i)
	{
		cv::Vec3i f = F[i];

		for(int j=0;j<3;j++){
			int ind = f[j];
			pt_src[j] = cp_src[ind];
			pt_dst[j] = cp_dst[ind];
		}

		warpTextureFromTriangle(pt_src, img_in, pt_dst, img_out);
	}
}
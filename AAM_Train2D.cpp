#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include "AAM_Train2D.h"

#include "dirent.h"
#include "basicFunctions.h"
#include "imageProcessing.h"
#include "delaunay2d.h"
//#include "warp_triangle_double.h"

void writeMat( cv::Mat const& mat, const char* filename, const char* varName = "A", bool bgr2rgb = true);

void load_contour_txt(const std::string filename, std::vector<cv::Point2f> &contour)
{
	char buf[255];
	float x, y;

	std::ifstream inFile(filename);
	if(!inFile.eof())
	{
		while (inFile.good())
		{
			inFile.getline(buf, 255);
			std::string line(buf);
			std::istringstream iss(line);
			iss >> x;
			iss >> y;
			contour.push_back(cv::Point2f(x,y));
		}
		inFile.close();
	}
}

int load_contour_yml(const std::string filename, std::vector<cv::Point2f> &contour)
{
	cv::FileStorage f;
	f.open(filename, cv::FileStorage::READ);
	assert(f.isOpened());

	cv::Mat C;
	f["Vertices"] >> C;
	C.convertTo(C, CV_64FC1);
	int cp_n = C.rows;
	int cp_dim = C.cols;

	contour = std::vector<cv::Point2f>(cp_n);
	for (int i=0; i<cp_n;i++){
		float x, y;
		x=(float)C.at<double>(i,1)-1;
		y=(float)C.at<double>(i,0)-1;
		contour[i] = cv::Point2f(x,y);
	}
	f.release();

	return cp_n;
}

void load_Data(std::string dir_shape, std::string dir_ims, AAMTrainingData &TrainingData)
{
	std::vector<std::string> shape_file_path_list;
	std::vector<std::string> ims_file_path_list;
	getImFiles(dir_shape,shape_file_path_list);
	getImFiles(dir_ims,ims_file_path_list);


	int n = shape_file_path_list.size();
	TrainingData.initialize(n);
	for(int i = 0;i<n;i++){
		TrainingData.Texture[i]=cv::imread(ims_file_path_list[i], CV_LOAD_IMAGE_GRAYSCALE);
		load_contour_yml(shape_file_path_list[i], TrainingData.Shape[i]);
	}
	TrainingData.n = TrainingData.Shape[0].size();
	return;
}

void vec_average(std::vector<cv::Point2f> &Vertices, cv::Point2f &offestsv)
{
	int n = Vertices.size();
	offestsv = cv::Point2f(0,0);
	for(int i=0;i<n;i++){
		offestsv.x+=Vertices[i].x;
		offestsv.y+=Vertices[i].y;
	}

	offestsv.x = offestsv.x/(float)n;
	offestsv.y = offestsv.y/(float)n;
}

void shift_scale_vec(std::vector<cv::Point2f> &VerticesIn, std::vector<cv::Point2f> &VerticesOut, cv::Point2f &shift, float scale)

{
	int n = VerticesIn.size();
	std::vector<cv::Point2f> VerticesOutT(n);
	for(int i=0;i<n;i++){
		VerticesOutT[i].x=(VerticesIn[i].x+shift.x)*scale;
		VerticesOutT[i].y=(VerticesIn[i].y+shift.y)*scale;
	}
	VerticesOut = VerticesOutT;

}

void scale_shift_vec(std::vector<cv::Point2f> &VerticesIn, std::vector<cv::Point2f> &VerticesOut, cv::Point2f &shift, float scale)

{
	int n = VerticesIn.size();
	std::vector<cv::Point2f> VerticesOutT(n);
	for(int i=0;i<n;i++){
		VerticesOutT[i].x=VerticesIn[i].x/scale-shift.x;
		VerticesOutT[i].y=VerticesIn[i].y/scale-shift.y;
	}
	VerticesOut= VerticesOutT;
}

void AAM_align_data_inverse2D_tire(std::vector<cv::Point2f> &VerticesIn, std::vector<cv::Point2f> &VerticesOut, AAMTform &T)
{
	scale_shift_vec(VerticesIn, VerticesOut, T.shift, T.scale);
}

void AAM_align_data2D_tire(std::vector<cv::Point2f> &Vertices,std::vector<cv::Point2f> &VerticesB, std::vector<cv::Point2f> &VerticesOut, AAMTform &tform)
// Remove rotation and translation and scale : Procrustes analysis 
{
	cv::Point2f offsetv, offsetvB;
	std::vector<cv::Point2f> VerticesS, VerticesSB;
	vec_average(Vertices, offsetv);
	offsetv = cv::Point2f(-offsetv.x, -offsetv.y);
	shift_scale_vec(Vertices, VerticesS, offsetv, 1.0);

	vec_average(VerticesB, offsetvB);
	offsetvB = cv::Point2f(-offsetvB.x, -offsetvB.y);
	shift_scale_vec(VerticesB, VerticesSB, offsetvB, 1.0);

	//  Set scaling to base example
	int n = Vertices.size();
	double d = 0;
	double dB = 0;
	for(int i=0;i<n;i++){
		d+=sqrt((pow((double)VerticesS[i].x,2.0)+pow((double)VerticesS[i].y,2.0)));
		dB+= sqrt((pow((double)VerticesSB[i].x,2.0)+pow((double)VerticesSB[i].y,2.0)));
	}

	d/=n;
	dB/=n;

	double offsets=dB/d;

	shift_scale_vec(VerticesS, VerticesOut, cv::Point2f(0,0), offsets);

	tform.shift = offsetv;
	tform.scale = offsets;
}

void sum_vertices(std::vector<cv::Point2f> &VerticesA, std::vector<cv::Point2f> &VerticesB)
{
	int n = VerticesA.size();
	assert(n==VerticesB.size());

	for(int i=0;i<n;i++){
		VerticesB[i].x = VerticesB[i].x + VerticesA[i].x;
		VerticesB[i].y = VerticesB[i].y + VerticesA[i].y;
	}
}

int pca_eigenvectors(cv::Mat &A, cv::Mat &eigenvalues, cv::Mat &eigenvectors, cv::Mat &psi)
{
	//mean(A,2)
	
	cv::reduce(A, psi, 2, CV_REDUCE_AVG);
	//writeMat(psi, "ei.mat", "ei");
	//writeMat(A, "a.mat", "a");
	int cols = A.cols;
	cv::Mat PsiM;
	cv::repeat(psi, 1, cols, PsiM);

	cv::Mat B, Bt;
	cv::subtract(A, PsiM, B);
	cv::transpose(B,Bt);
	cv::Mat L=Bt*B;
	
	cv::eigen(L, eigenvalues, eigenvectors);
	cv::transpose(eigenvectors,eigenvectors);
	
	eigenvectors = B*eigenvectors;
	cv::Mat temp = cv::Mat::zeros(eigenvalues.size(), eigenvalues.type());
	cv::scaleAdd(eigenvalues, 1.0/double(cols-1), temp, eigenvalues); 

	// Normalize Vectors to unit length, kill vectors corr. to tiny evalues
	int num_good = 0;
	int numvecs=0;
	double eps = 0.00001;
	for(int i=0;i<cols;i++){
		cv::Mat vec = eigenvectors.col(i);
		if(eigenvalues.at<double>(i,0) < eps){
			vec = cv::Mat::zeros(vec.size(), vec.type());
			eigenvalues.at<double>(i,0)=0;
		}
		else{
			cv::normalize(vec, vec);
			num_good++;
		}

		if(eigenvalues.at<double>(i,0) > 0){
			numvecs++;
		}
	}
	
	//std::cout << eigenvectors << std::endl;
	return num_good;
}

void load_triangulation(std::string filename, cv::Mat &F)
{
	cv::FileStorage f;
	f.open(filename, cv::FileStorage::READ);
	assert(f.isOpened());
	f["tri"] >> F;
	F=F-1;//shift to correct to c++ indexing
	f.release();
}

void load_triangulation(std::string filename, std::vector<cv::Vec3i> &F)
{
	cv::FileStorage f;
	f.open(filename, cv::FileStorage::READ);
	assert(f.isOpened());
	cv::Mat temp;
	f["tri"] >> temp;

	F = std::vector<cv::Vec3i> (temp.rows);
	for(int i=0;i<temp.rows;i++){
		for(int j=0;j<temp.cols;j++){
			F[i][j]=(int)temp.at<float>(i,j)-1;//shift to correct to c++ indexing
		}
	}

	f.release();
}

void AAM_MakeShapeModel2D_tire(AAMTrainingData &TrainingData, AAMShape &ShapeModel, AMM_Model2D_Options &options)
{
	// Number of datasets
	int s=TrainingData.N;

	// Number of landmarks
	int nl = TrainingData.n;

	// Shape model
	// Remove rotation and translation and scale : Procrustes analysis 
	std::vector<cv::Point2f> MeanVertices=TrainingData.Shape[0];
	std::vector<cv::Point2f> VerticesC;
	
	cv::Point2f Sh;
	double Sc;
	std::vector<AAMTrainingData> TrainingDataS;
	for(int k=0;k<2;k++){
		Sc = 0;
		Sh = cv::Point2f(0,0);
		std::vector<cv::Point2f> AllVertices(nl,cv::Point2f(0,0));
		for(int i=0;i<s;i++)
		{
			AAMTform tform;
			AAM_align_data2D_tire(TrainingData.Shape[i], MeanVertices, VerticesC, tform);
			TrainingData.Shift[i]=tform.shift;
			TrainingData.Scale[i]=tform.scale;
			sum_vertices(VerticesC, AllVertices);
			Sh.x += tform.shift.x;
			Sh.y += tform.shift.y;
			Sc += tform.scale;
		}

		cv::Point2f temp(0,0);
		shift_scale_vec(AllVertices, AllVertices, temp, 1.0/(double)s);
		Sh.x /=s;
		Sh.y /=s;
		Sc/=s;
		AAMTform T;
		T.scale = Sc;
		T.shift = Sh;
		AAM_align_data_inverse2D_tire(AllVertices, MeanVertices, T);
	}
	//writeMat(cv::Mat(MeanVertices), "cp.mat", "cp"); 
	
	for(int i=0;i<s;i++){
		AAMTform tform;
		AAM_align_data2D_tire(TrainingData.Shape[i], MeanVertices, TrainingData.ShapeC[i], tform);
		TrainingData.Shift[i]=tform.shift;
		TrainingData.Scale[i]=tform.scale;
	}

	// Construct a matrix with all contour point data of the training data set
	cv::Mat X(nl*2, s, CV_64FC1);
	for(int q=0;q<s;q++){
		for(int i=0;i<nl;i++){
			X.at<double>(i,q)=TrainingData.ShapeC[q][i].x;
			X.at<double>(i+nl,q)=TrainingData.ShapeC[q][i].y;
		}
	}
	//writeMat(X, "X.mat", "X1"); 

	//PCA
	cv::Mat eigenvalues;
	cv::Mat eigenvectors;
	cv::Mat psi;
	pca_eigenvectors(X, eigenvalues, eigenvectors, psi);
	//writeMat(psi, "ei.mat", "ei"); 
	
	// Keep only 98% of all eigen vectors, (remove contour noise)
	std::vector<double> eig_cumsum(s);
	double c_sum = eigenvalues.at<double>(0,0);
	
	for(int i=0;i<s;i++){
		eig_cumsum[i]=c_sum;
	
		if(i<s-1){
			c_sum+=eigenvalues.at<double>(i+1,0);
		}
	}

	int ind=0;
	double ei_sum=eig_cumsum[s-1]*0.99;
	for(int i=0;i<s;i++){
		if(eig_cumsum[i]>ei_sum){
			ind=i;
			break;
		}
	}

	eigenvalues = eigenvalues(cv::Range(0, ind+1), cv::Range(0, 1));
	eigenvectors = eigenvectors(cv::Range(0, nl*2), cv::Range(0, ind+1));

	ShapeModel.Evalues = eigenvalues;
	ShapeModel.Evectors = eigenvectors;
	ShapeModel.data_mean = psi;
	ShapeModel.data = X;

	double mu, var;

	basicStat(TrainingData.Scale, mu, var, 0, s-1);
	ShapeModel.varSc=var;
	ShapeModel.MeanVertices = MeanVertices;

	// Build Delaunay
	/*
	std::vector<cv::Point2f> tempC(TrainingData.n);
	for(int i=0;i<TrainingData.n;i++){
		tempC[i]=cv::Point2f(TrainingData.Shape[0][i]);
	}
	cv::Subdiv2D subdiv;
	build_delaunay(TrainingData.Texture[0], tempC, subdiv);
	build_tri_face_data(subdiv, tempC, ShapeModel.F, TrainingData.Texture[ind].size());
	*/

	double minVal, maxVal;
	cv::minMaxLoc(ShapeModel.data_mean, &minVal, &maxVal);
	minVal = minVal*-1.0;
	#undef max
	int ts = (int)my_round(std::max(maxVal, minVal))*2*options.texturesize;
	ShapeModel.textureSize = cv::Size(ts,ts);

	return;
	//writeMat(cv::Mat(eig_cumsum), "ei.mat", "ei"); 
	//int yj=0;
	
	/*
	ShapeData.Lines = TrainingData(1).Lines;
	ts=ceil(max(max(ShapeData.x_mean(:)),-min(ShapeData.x_mean(:)))*2*options.texturesize);
	ShapeData.TextureSize=[ts ts];
	ShapeData.Tri= delaunay(x_mean(1:end/2),x_mean(end/2+1:end));
	*/
}

void drawObject(cv::Size &imsize, cv::Mat &mask, std::vector<cv::Point2f> &base_points)
{
	cv::Mat temp = cv::Mat::zeros(imsize, CV_8UC1);
	std::vector<std::vector<cv::Point> > tC(1);
	int n = base_points.size();
	for(int i=0;i<n;i++){
		tC[0].push_back(cv::Point(base_points[i]));
	}
	cv::drawContours(temp, tC, 0, cv::Scalar(255), -1);
	mask = temp;
	//cv::imshow("test", temp);cv::waitKey();
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

template <class T>
void crop_mask(cv::Mat &img, cv::Mat &mask, int k, cv::Mat &out)
{
	int m = img.rows;
	int n = img.cols;

	out = cv::Mat_ <T> (k, 1);
	int s=0;
	for(int j=0;j<n;j++){
		for(int i=0;i<m;i++){
			if(mask.at<uchar>(i,j) > 0){
				out.at<T>(s,0) = img.at<T>(i,j);
				s++;
			}
		}
	}
}

void vec3i2mat(std::vector<cv::Vec3i> &F, cv::Mat &out)
{
	int n = F.size();
	out = cv::Mat (n, 3, CV_64FC1);
	
	for(int i=0;i<n;i++){
		out.at<double>(i,0) = F[i][0];
		out.at<double>(i,1) = F[i][1];
		out.at<double>(i,2) = F[i][2];
	}
}

void AAM_Appearance2Vector2D(cv::Mat &in, cv::Size &ts, int k, std::vector<cv::Point2f> &source, std::vector<cv::Point2f> &target, 
							 std::vector<cv::Vec3i> &F/*cv::Mat &F*/, cv::Mat &greyvector, cv::Mat &mask, bool draw_flag=false)
{
	/*
	cv::Mat out=cv::Mat::zeros(ts, CV_64FC1);
	double *Iout = (double*)out.data; 
	int sizeIout[2] = {out.rows, out.cols};

	cv::Mat tempin;in.convertTo(tempin, CV_64FC1);
	//tempin.t();
	double *Iin = (double*)tempin.data; 
	int sizeIin[3] = {tempin.rows, tempin.cols, 1};

	cv::Mat xy;
	contour2mat(source, xy);
	xy.convertTo(xy, CV_64FC1);
	//xy.t();
	double *XY = (double*)xy.data; 
	int sizeXY[2] = {source.size(), 2};

	cv::Mat uv;
	contour2mat(target, uv);
	uv.convertTo(uv, CV_64FC1);
	//uv.t();
	double *UV = (double*)uv.data; 
	int sizeUV[2]= {target.size(), 2}; 

	cv::Mat tri;
	//vec3i2mat(F, tri);
	F.convertTo(tri, CV_64FC1);
	//tri.t();
	double *TRI =(double*)tri.data; 
	int sizeTRI[2] = {target.size(), 3}; 

	writeMat(xy, "a.mat", "a", false);
	warp_triangle_double(Iout, sizeIout, Iin, sizeIin, XY, sizeXY, UV, sizeUV, TRI, sizeTRI);

	cv::Mat drawing;
	imageplot(out, drawing);
	cv::imshow("test", drawing);waitKey();
	*/
	cv::Mat out = cv::Mat::zeros(ts, CV_8UC1);
	wrap_piecewise_nonlin(in, out, F, source, target);
	if(draw_flag==true){
		cv::imshow("test", out);waitKey();
	}
	crop_mask<uchar>(out, mask, k, greyvector);

}

void AAM_NormalizeAppearance2D(cv::Mat &gim)
{
	//Normalize appearance data grey values
	cv::Mat mu, su;
	cv::meanStdDev(gim, mu, su);
	cv::subtract(gim, mu, gim);
	cv::Mat temp(gim.size(), CV_64FC1);
	cv::scaleAdd(gim, 1.0/su.at<double>(0,0), temp, gim);
}
void AAM_MakeAppearanceModel2D(AAMTrainingData &TrainingData, AAMShape &ShapeModel, AAMAppearance &AppearanceData, AMM_Model2D_Options &options)
{
	// Coordinates of mean contour
	std::vector<cv::Point2f> base_points(TrainingData.n);
	for(int i=0;i<TrainingData.n;i++){
		base_points[i].x = (float)ShapeModel.data_mean.at<double>(i,0);
		base_points[i].y = (float)ShapeModel.data_mean.at<double>(i+TrainingData.n,0);
	} 
	// Normalize the base points to range 0..1
	float minX, maxX,  minY, maxY;
	MinMaxContour(base_points, minX, maxX,  minY, maxY);
	for(int i=0;i<TrainingData.n;i++){
		base_points[i].x = (base_points[i].x-minX);
		base_points[i].y = (base_points[i].y-minY);
	}
	MinMaxContour(base_points, minX, maxX,  minY, maxY);
	for(int i=0;i<TrainingData.n;i++){
		base_points[i].x /= maxX;
		base_points[i].y /= maxY;
		//Transform the mean contour points into the coordinates in the texture image.
		base_points[i].x=1+(ShapeModel.textureSize.width-1)*base_points[i].x;
		base_points[i].y=1+(ShapeModel.textureSize.height-1)*base_points[i].y;
	}
	//writeMat(cv::Mat(base_points), "cp.mat", "cp");
	//writeMat(ShapeModel.data_mean, "cp.mat", "cp"); 
	
	cv::Mat mask;
	drawObject(ShapeModel.textureSize, mask, base_points);

	int k=cv::countNonZero(mask);
	
	cv::Mat grey(k, TrainingData.N, CV_64FC1);
	for(int i=0;i<TrainingData.N;i++){
		std::cout << i << std::endl;
		cv::Mat greyvector;
		AAM_Appearance2Vector2D(TrainingData.Texture[i], ShapeModel.textureSize, k, TrainingData.Shape[i], base_points, ShapeModel.F, greyvector, mask);
		cv::Mat temp = grey(cv::Range(0, k), cv::Range(i,i+1));
		greyvector.convertTo(temp, CV_64FC1);
		AAM_NormalizeAppearance2D(temp);
	}

	//PCA
	cv::Mat eigenvalues;
	cv::Mat eigenvectors;
	cv::Mat psi;
	pca_eigenvectors(grey, eigenvalues, eigenvectors, psi);
	//writeMat(psi, "ei.mat", "ei"); 
	
	// Keep only 99% of all eigen vectors, (remove contour noise)
	std::vector<double> eig_cumsum(TrainingData.N);
	double c_sum = eigenvalues.at<double>(0,0);
	
	for(int i=0;i<TrainingData.N;i++){
		eig_cumsum[i]=c_sum;
	
		if(i<TrainingData.N-1){
			c_sum+=eigenvalues.at<double>(i+1,0);
		}
	}

	int ind=0;
	double ei_sum=eig_cumsum[TrainingData.N-1]*0.99;
	for(int i=0;i<TrainingData.N;i++){
		if(eig_cumsum[i]>ei_sum){
			ind=i;
			break;
		}
	}

	eigenvalues = eigenvalues(cv::Range(0, ind+1), cv::Range(0, 1));
	eigenvectors = eigenvectors(cv::Range(0, eigenvectors.rows), cv::Range(0, ind+1));
	
	// Store the Eigen Vectors and Eigen Values
	AppearanceData.k = k;
	AppearanceData.Evectors=eigenvectors;
	AppearanceData.Evalues=eigenvalues;
	AppearanceData.g_mean=psi;
	AppearanceData.g = grey;
	AppearanceData.ObjectPixels=mask;
	AppearanceData.base_points=base_points;
}

void AAM_Weights2D_tire(AAMTrainingData &TrainingData, AAMShape &ShapeData, AAMAppearance &AppearanceData, AMM_Model2D_Options &options)
{
	int N = TrainingData.N;
	int n = TrainingData.n;

	cv::Mat Change = cv::Mat::zeros(N, ShapeData.Evectors.cols, CV_64FC1);

	for(int i=0;i<N;i++)
	{
		//Remove translation and rotation, as done when training the model.
		std::vector<cv::Point2f> pos;
		AAMTform tform;
		AAM_align_data2D_tire(TrainingData.Shape[i], ShapeData.MeanVertices, pos, tform);
		
		// Describe the model by a vector b with model parameters
		cv::Mat X(n*2, 1, CV_64FC1);
		for(int i=0;i<n;i++){
			X.at<double>(i,0)=(double)pos[i].x;
			X.at<double>(i+n,0)=(double)pos[i].y;
		}

		cv::subtract(X, ShapeData.data_mean, X);
		cv::Mat b = ShapeData.Evectors.t()*X;

		//std::cout << b << std::endl;

		// Get the intensities of the untransformed shape.
		// Because noisy eigenvectors from the shape were removed, the 
		// contour is on a little different position and
		// intensities probabbly differ a little bit from the orignal appearance
		cv::Mat x_normal = ShapeData.data_mean + ShapeData.Evectors*b;
		
		std::vector<cv::Point2f> pos_normal(n);
		for(int i=0;i<n;i++){
			pos_normal[i].x=(float)x_normal.at<double>(i,0);
			pos_normal[i].y=(float)x_normal.at<double>(i+n,0);
		}
		
		AAM_align_data_inverse2D_tire(pos_normal, pos_normal, tform);
		cv::Mat g_normal;
		AAM_Appearance2Vector2D(TrainingData.Texture[i], ShapeData.textureSize, AppearanceData.k, 
			pos_normal, AppearanceData.base_points, ShapeData.F, g_normal, AppearanceData.ObjectPixels, true);
		g_normal.convertTo(g_normal, CV_64FC1);
		AAM_NormalizeAppearance2D(g_normal);

		double K[2] = {-0.5, 0.5};
		for (int j = 0; j<ShapeData.Evectors.cols; j++){
            for(int k=0; k<1; k++){

                // Change on model parameter a little bit, to see the influence
                // from the shape parameters on appearance parameters
                cv::Mat b_offset=b.clone();  
				b_offset.at<double>(j,0)=b_offset.at<double>(j,0)+K[k];

                // Transform the model parameter vector b , back to contour positions
                cv::Mat x_offset= ShapeData.data_mean + ShapeData.Evectors*b_offset;
				std::vector<cv::Point2f> pos_offset(n);
				for(int i=0;i<n;i++){
					pos_offset[i].x=(float)x_offset.at<double>(i,0);
					pos_offset[i].y=(float)x_offset.at<double>(i+n,0);
				}
                
                // Now add the previously removed translation and rotation
				AAM_align_data_inverse2D_tire(pos_offset, pos_offset, tform);
				cv::Mat g_offset;

				AAM_Appearance2Vector2D(TrainingData.Texture[i], ShapeData.textureSize, AppearanceData.k, 
					pos_offset, AppearanceData.base_points, ShapeData.F, g_offset, AppearanceData.ObjectPixels, true);
				g_offset.convertTo(g_offset, CV_64FC1);
				AAM_NormalizeAppearance2D(g_offset);

				double s=0;
				for(int h=0;h<AppearanceData.k;h++){
					s+=pow((g_offset.at<double>(h,0)-g_normal.at<double>(h,0)),2.0);
				}
				s/=AppearanceData.k;
				s=sqrt(s);
                Change.at<double>(i,j) = Change.at<double>(i,j)+s;
			}
		}
	}

	cv::Mat Ws=cv::Mat::zeros(ShapeData.Evectors.cols,ShapeData.Evectors.cols, CV_64FC1);
	for (int j = 0;j<ShapeData.Evectors.cols;j++){
		cv::Scalar mu = mean(Change.col(j));
		Ws.at<double>(j,j) = mu[0];
	}
}

void AAM_CombineShapeAppearance2D_tire(AAMTrainingData &TrainingData, AAMShape &ShapeData, AAMAppearance &AppearanceData, AMM_Model2D_Options &options)
{
	//This functions combines the shape and appearance of the objects, by
	//adding the weighted vector describing shape and appearance, followed by
	//PCA

	// Get weight matrix. The Weights are a scaling between texture and shape
	//to give a change in shape parameters approximately the same 
	//influences as texture parameters.
		
	//Ws = AAM_Weights2D_tire(TrainingData,ShapeData,AppearanceData,options);
}
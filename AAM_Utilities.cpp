#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "dirent.h"
#include "AAM_Utilities.h"

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

int load_contour_yml(const std::string filename, cv::Mat &contour)
{
	cv::FileStorage f;
	f.open(filename, cv::FileStorage::READ);
	assert(f.isOpened());

	cv::Mat C;
	f["Vertices"] >> C;
	C.convertTo(C, CV_64FC1);
	C = C - 1.0;
	int cp_n = C.rows;
	int cp_dim = C.cols;

	contour = cv::Mat(cp_n, 2, CV_64FC1);
	//cv::Mat y = contour(cv::Range(0, cp_n), cv::Range::all());
	//cv::Mat x = contour(cv::Range(cp_n, 2*cp_n), cv::Range::all());
	
	C.col(0).copyTo(contour.col(1));
	C.col(1).copyTo(contour.col(0));
	f.release();

	return cp_n;
}

void load_Data(std::string dir_shape, std::string dir_ims, AAMTrainingData &TrainingData)
{
	std::vector<std::string> shape_file_path_list;
	std::vector<std::string> ims_file_path_list;
	getImFiles(dir_shape,shape_file_path_list);
	getImFiles(dir_ims,ims_file_path_list);


	//int n = shape_file_path_list.size();
	int n = 5;
	TrainingData.initialize(n);
	for(int i = 0;i<n;i++){
		TrainingData.Texture[i]=cv::imread(ims_file_path_list[i], CV_LOAD_IMAGE_GRAYSCALE);
		load_contour_yml(shape_file_path_list[i], TrainingData.Shape[i]);
	}
	TrainingData.n = TrainingData.Shape[0].rows;
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

void shift_scale_vec(std::vector<cv::Point2f> &VerticesIn, std::vector<cv::Point2f> &VerticesOut, 
					 cv::Point2f &shift, float scale)

{
	int n = VerticesIn.size();
	std::vector<cv::Point2f> VerticesOutT(n);
	for(int i=0;i<n;i++){
		VerticesOutT[i].x=(VerticesIn[i].x+shift.x)*scale;
		VerticesOutT[i].y=(VerticesIn[i].y+shift.y)*scale;
	}
	VerticesOut = VerticesOutT;

}

void scale_shift_vec(std::vector<cv::Point2f> &VerticesIn, 
					 std::vector<cv::Point2f> &VerticesOut, cv::Point2f &shift, float scale)

{
	int n = VerticesIn.size();
	std::vector<cv::Point2f> VerticesOutT(n);
	for(int i=0;i<n;i++){
		VerticesOutT[i].x=VerticesIn[i].x/scale-shift.x;
		VerticesOutT[i].y=VerticesIn[i].y/scale-shift.y;
	}
	VerticesOut= VerticesOutT;
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

void load_triangulation(std::string filename, cv::Mat &F)
{
	cv::FileStorage f;
	f.open(filename, cv::FileStorage::READ);
	assert(f.isOpened());
	f["tri"] >> F;
	F.convertTo(F, CV_64FC1);
	F=F-1.0;//shift to correct to c++ indexing
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

void sum_vertices(std::vector<cv::Point2f> &VerticesA, std::vector<cv::Point2f> &VerticesB)
{
	int n = VerticesA.size();
	assert(n==VerticesB.size());

	for(int i=0;i<n;i++){
		VerticesB[i].x = VerticesB[i].x + VerticesA[i].x;
		VerticesB[i].y = VerticesB[i].y + VerticesA[i].y;
	}
}

void pca_eigenvectors_svd(cv::Mat &A, cv::Mat &eigenvalues, cv::Mat &eigenvectors, cv::Mat &psi)
{
	// Calculate the mean 
	cv::reduce(A, psi, 2, CV_REDUCE_AVG);
	
	//Substract the mean
	int cols = A.cols;
	cv::Mat PsiM;
	cv::repeat(psi, 1, cols, PsiM);

	cv::Mat B;
	cv::subtract(A, PsiM, B);
	B = B*1.0/((double)cols-1);

	//Compute SVD
	cv::Mat S, U, V;
	cv::SVD::compute(B, S, U, V);
	
	eigenvectors = U;
	eigenvalues=S;
	//cv::pow(eigenvalues, 2.0, eigenvalues);

	//Set sign according to first element of each eigenvector
	for(int i=0;i<eigenvectors.cols;i++){
		cv::Mat vec = eigenvectors.col(i);
		if(vec.at<double>(0,0)<0){
			vec=(-1.0)*vec;
		}
	}

	//writeMat(eigenvalues, "ei.mat", "ei");

}

void pca_eigenvectors_dir(cv::Mat &A, cv::Mat &eigenvalues, cv::Mat &eigenvectors, cv::Mat &psi)
{
	int numcomponents = A.cols;
	cv::PCA pca(A, // pass the data
		cv::Mat(), // there is no pre-computed mean vector,
		// so let the PCA engine to compute it
		CV_PCA_DATA_AS_COL, // indicate that the vectors
		// are stored as matrix cols
		// (use CV_PCA_DATA_AS_ROW if the vectors are
		// the matrix rows)
		numcomponents // specify how many principal components to retain
		);

	// And copy the PCA results:
	psi = pca.mean.clone();
	eigenvalues = pca.eigenvalues.clone();
	eigenvectors = pca.eigenvectors.clone();
	eigenvectors=eigenvectors.t();

	for(int i=0;i<eigenvectors.cols;i++){
		cv::Mat vec = eigenvectors.col(i);
		if(vec.at<double>(0,0)<0){
			vec=(-1.0)*vec;
		}
	}

	//writeMat(eigenvalues, "ei.mat", "ei");
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

	//writeMat(eigenvectors, "a.mat", "a");
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

	for(int i=0;i<eigenvectors.cols;i++){
		cv::Mat vec = eigenvectors.col(i);
		if(vec.at<double>(0,0)<0){
			vec=(-1.0)*vec;
		}
	}

	//writeMat(eigenvalues, "ei.mat", "ei");
	//std::cout << eigenvectors << std::endl;
	return num_good;
}

void get_components(cv::Mat &data, int numcomponents, cv::Mat &mean, cv::Mat &eigenvalues, cv::Mat &eigenvectors)
{
	pca_eigenvectors(data, eigenvalues, eigenvectors, mean);

	// Keep only 98% of all eigen vectors, (remove contour noise)
	std::vector<double> eig_cumsum(numcomponents);
	double c_sum = eigenvalues.at<double>(0,0);

	for(int i=0;i<numcomponents;i++){
		eig_cumsum[i]=c_sum;

		if(i<numcomponents-1){
			c_sum+=eigenvalues.at<double>(i+1,0);
		}
	}

	int ind=0;
	double ei_sum=eig_cumsum[numcomponents-1]*0.99;
	for(int i=0;i<numcomponents;i++){
		if(eig_cumsum[i]>ei_sum){
			ind=i;
			break;
		}
	}

	eigenvalues = eigenvalues(cv::Range(0, ind+1), cv::Range::all());
	eigenvectors = eigenvectors(cv::Range::all(), cv::Range(0, ind+1));
}

void saveAAMData(AAM_ALL_DATA &Data, std::string filename)
{
	// Eigen Vectors and Eigen Values
	cv::Mat MeanVertices;
	cv::FileStorage f;
	f.open(filename, cv::FileStorage::WRITE);
	
	//Shape
	f << "Sevec" << Data.S.Evectors;
	f << "Seval" << Data.S.Evalues;
	f << "Smean" << Data.S.data_mean;
	f << "Sdata" << Data.S.data;
	f << "Sver" << Data.S.MeanVertices;

	//Appearance
	f << "Aevec" << Data.A.Evectors;
	f << "Aeval" << Data.A.Evalues;
	f << "Amean" << Data.A.g_mean;
	f << "Adata" << Data.A.g;
	f << "Aver" << Data.A.base_points;
	f << "Amask" << Data.A.ObjectPixels;
	f << "k" << Data.A.k;

	//Shape-Appearance
	f << "SAevec" << Data.SA.Evectors;
	f << "SAeval" << Data.SA.Evalues;
	f << "SAmean" << Data.SA.b_mean;
	f << "SAdata" << Data.SA.b;
	f << "SAws" << Data.SA.Ws;

	f << "R" << Data.R;

	f << "n" << Data.n;
	f << "N" << Data.N;

}

void loadAAMData(AAM_ALL_DATA &Data, std::string filename)
{
	// Eigen Vectors and Eigen Values
	cv::Mat MeanVertices;
	cv::FileStorage f;
	f.open(filename, cv::FileStorage::READ);
	
	//Shape
	f["Sevec"] >> Data.S.Evectors;
	f["Seval"] >> Data.S.Evalues;
	f["Smean"] >> Data.S.data_mean;
	f["Sdata"] >> Data.S.data;
	f["Sver"] >> Data.S.MeanVertices;

	//Appearance
	f["Aevec"] >> Data.A.Evectors;
	f["Aeval"] >> Data.A.Evalues;
	f["Amean"] >> Data.A.g_mean;
	f["Adata"] >> Data.A.g;
	f["Aver"] >> Data.A.base_points;
	f["Amask"] >> Data.A.ObjectPixels;
	f["k"] >> Data.A.k;

	//Shape-Appearance
	f["SAevec"] >> Data.SA.Evectors;
	f["SAeval"] >> Data.SA.Evalues;
	f["SAmean"] >> Data.SA.b_mean;
	f["SAdata"] >> Data.SA.b;
	f["SAws"] >> Data.SA.Ws;

	f["R"] >> Data.R;

	f["n"] >> Data.n;
	f["N"] >> Data.N;
}

void saveAAMTextue(AAMTexture &Text, std::string filename)
{
	// Eigen Vectors and Eigen Values
	cv::Mat MeanVertices;
	cv::FileStorage f;
	f.open(filename, cv::FileStorage::WRITE);
	
	//Texture
	f << "F" << Text.F;
	f << "rows" << Text.textureSize.height;
	f << "cols" << Text.textureSize.width;
}

void loadAAMTexture(AAMTexture &Text, std::string filename)
{
	// Eigen Vectors and Eigen Values
	cv::Mat MeanVertices;
	cv::FileStorage f;
	f.open(filename, cv::FileStorage::READ);
	
	//Texture
	f["F"] >> Text.F;
	f["rows"] >> Text.textureSize.height;
	f["cols"] >> Text.textureSize.width;
}

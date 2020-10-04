#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <fstream>

#include "mail.h"
#include "io_disp.h"
//#include "io_flow.h" // no flow evaluation here;
#include "io_integer.h"
#include "utils.h"
//added by CCJ
#include "../pgm_pfm/pfm_rw.h"

using namespace std;
#include <boost/python.hpp>
#include "boost/python/extract.hpp"

//#include "boost/python/numeric.hpp"
/* Updated by CCJ for Boost version 1.65:
 * Boost 1.65 removes boost/python/numeric.hpp 
 */
// BOOST_LIB_VERSION and BOOST_VERSION are defined in this header file;
#include <boost/version.hpp> 
//#if BOOST_VERSION >= 106300 // >= 1.63.0
#include "boost/python/numpy.hpp"
namespace np = boost::python::numpy;
//#else
//#include "boost/python/numeric.hpp"
//namespace np = boost::python::numeric;
//#endif

#include <numpy/ndarrayobject.h>

#define isDisplay false
#define isShowErrorMap false
#define NUM_TEST_IMAGES 200
#define NUM_ERROR_IMAGES 200
#define ABS_THRESH 3.0
#define REL_THRESH 0.05

using namespace boost::python;
//const std::string baseDir = "/home/ccj/PKLS/";
const std::string baseDir = "./";
std::string imgDir;


std::string disp_suffix = "_sgm_disp0PKLS";

const int thred_3_idx = 4; // threds = {1a,1b, 2a,2b, 3a, 3b, 4a, 4b, 5a, 5b}

//****************
// declaration:
//****************
vector<float> disparityErrorsOutlier_kt15 (DisparityImage &D_gt,DisparityImage &D_orig,DisparityImage &D_ipol,
		IntegerImage &O_map, //object map (0:background, >0:foreground)
		bool is_mae_rmse = false
		);

vector<float> disparityErrorsOutlier (DisparityImage &D_gt,DisparityImage &D_orig,DisparityImage &D_ipol,	bool refl);

vector<float> disparityErrorsAverage (DisparityImage &D_gt,	DisparityImage &D_orig, DisparityImage &D_ipol,bool refl);

void pfm2uint16PNG(	string pfm_result_file, string disp_suf, 	string png_result_file,	int imgNum); 
bool eval (string result_sha,string evaluate, float * err_result,  int imgIdxStart, int imgIdxEnd, int kt2012_2015_type, Mail* mail );
bool resultsAvailable (const string & result_sha, const int & imgIdxStart,	const int & imgIdxEnd,	Mail * mail = NULL 	);
bool resultsAvailable(const string & result_dir, std::vector<std::string> & v_imgs);


//*****************************
//*** disparity, kt 2015 ******
//*****************************
vector<float> disparityErrorsOutlier_kt15 (
		DisparityImage &D_gt,
		DisparityImage &D_orig,
		DisparityImage &D_ipol,
		IntegerImage &O_map, //object map (0:background, >0:foreground)
    bool is_mae_rmse // metrics of mas and rmse, added on 2019/08/30
		){

  // check file size
  if (D_gt.width()!=D_orig.width() || D_gt.height()!=D_orig.height()) {
		printf("D_gt.width=%d, D_orig.width=%d, D_gt.height=%d, D_orig.height=%d",
				D_gt.width(), D_orig.width(), D_gt.height(), D_orig.height());
    cout << "ERROR: KT15 Wrong file size!" << endl;
    throw 1;
  }

  // extract width and height
  int32_t width  = D_gt.width();
  int32_t height = D_gt.height();

  // init errors
  vector<float> errors;
  int32_t num_errors_bg = 0; // nubmer of outliers in bg
  int32_t num_pixels_bg = 0;// number of pixels in bg
  int32_t num_errors_bg_result = 0;
  int32_t num_pixels_bg_result = 0;
  int32_t num_errors_fg = 0;
  int32_t num_pixels_fg = 0;
  int32_t num_errors_fg_result = 0;
  int32_t num_pixels_fg_result = 0;
  int32_t num_errors_all = 0;
  int32_t num_pixels_all = 0;
  int32_t num_errors_all_result = 0;
  int32_t num_pixels_all_result = 0;
  

  // init errors for mae (mean absolute error) and rmse (root mean square error)
  vector<float> errors_mae_rmse = {
    .0, // bg, absolute error
    .0, // fg, absolute error
    .0, // all, absolute error
    .0, // bg, square error
    .0, // fg, square error
    .0  // all, square error
    };

  // for all pixels do
  for (int32_t u=0; u<width; u++) {
    for (int32_t v=0; v<height; v++) {
      if (D_gt.isValid(u,v)) {
        float d_gt  = D_gt.getDisp(u,v);
        float d_est = D_ipol.getDisp(u,v);
				// both absolute error and relative error;
        float d_errVal = fabs(d_gt-d_est);
        bool  d_err = fabs(d_gt-d_est)>ABS_THRESH && fabs(d_gt-d_est)/fabs(d_gt)>REL_THRESH;
        

				if (O_map.getValue(u,v)==0){//  0: background (bg);
          if (d_err){
						num_errors_bg++;
						errors_mae_rmse[0] += d_errVal;
						errors_mae_rmse[3] += d_errVal * d_errVal;
          }
          num_pixels_bg++;
          if (D_orig.isValid(u,v)) {// your calculated disparity
            if (d_err)
              num_errors_bg_result++;
            num_pixels_bg_result++;
          }
        }// end of bg;
				
				else { // > 0: foreground (fg);
          if (d_err){
            num_errors_fg++;
						errors_mae_rmse[1] += d_errVal;
						errors_mae_rmse[4] += d_errVal * d_errVal;
					}
          num_pixels_fg++;
          if (D_orig.isValid(u,v)) {
            if (d_err)
              num_errors_fg_result++;
            num_pixels_fg_result++;
          }
        }

        if (d_err){
          num_errors_all++;
					errors_mae_rmse[2] += d_errVal;
					errors_mae_rmse[5] += d_errVal * d_errVal;
				}
        num_pixels_all++;
        if (D_orig.isValid(u,v)){
          if (d_err)
            num_errors_all_result++;
          num_pixels_all_result++;
        }

      }// end of valid pixels with ground truth
    }
	}
  
	// push back errors and pixel count
  errors.push_back(num_errors_bg);
  errors.push_back(num_pixels_bg);
  errors.push_back(num_errors_bg_result);
  errors.push_back(num_pixels_bg_result);
  errors.push_back(num_errors_fg);
  errors.push_back(num_pixels_fg);
  errors.push_back(num_errors_fg_result);
  errors.push_back(num_pixels_fg_result);
  errors.push_back(num_errors_all);
  errors.push_back(num_pixels_all);
  errors.push_back(num_errors_all_result);
  errors.push_back(num_pixels_all_result);
  
	if (is_mae_rmse){
		for (int i = 0; i < 6; i++)
			errors.push_back(errors_mae_rmse[i]);
	}

	// push back density
  errors.push_back((float)num_pixels_all_result/max((float)num_pixels_all,1.0f));

  // return errors
  return errors;
}




//***********************************
//*** disparity error, kt 2012 ******
//***********************************
vector<float> disparityErrorsOutlier (
		DisparityImage &D_gt,
		DisparityImage &D_orig,
		DisparityImage &D_ipol,
		bool refl) {

  // check file size
  if (D_gt.width()!=D_orig.width() || D_gt.height()!=D_orig.height()) {
    cout << "ERROR: Wrong file size!" << endl;
    throw 1;
  }

  // extract width and height
  int32_t width  = D_gt.width();
  int32_t height = D_gt.height();

  // init errors
  vector<float> errors;
  for (int32_t i=0; i<2*5; i++)
    errors.push_back(0);
  int32_t num_pixels = 0;
  int32_t num_pixels_result = 0;

  // for all pixels do
  for (int32_t u=0; u<width; u++) {
    for (int32_t v=0; v<height; v++) {
      if (D_gt.isValid(u,v)) {
        float d_err  = fabs(D_gt.getDisp(u,v)-D_ipol.getDisp(u,v));
        for (int32_t i=0; i<5; i++)
          if (d_err>(float)(i+1))
            errors[i*2+0]++;
        num_pixels++;
        if (D_orig.isValid(u,v)) {
          for (int32_t i=0; i<5; i++)
            if (d_err>(float)(i+1))
              errors[i*2+1]++;
          num_pixels_result++;
        }
      }
    }
  }
  
  if (refl) {
  
    // push back counts
    errors.push_back((float)num_pixels);
    errors.push_back((float)num_pixels_result);
    
  } else {
  
    // check number of pixels
    if (num_pixels==0) {
      cout << "ERROR: Ground truth defect => Please write me an email!" << endl;
      throw 1;
    }

    // normalize errors
    for (int32_t i=0; i<errors.size(); i+=2)
      errors[i] /= max((float)num_pixels,1.0f);
    if (num_pixels_result>0)
      for (int32_t i=1; i<errors.size(); i+=2)
        errors[i] /= max((float)num_pixels_result,1.0f);

    // push back density
    errors.push_back((float)num_pixels_result/max((float)num_pixels,1.0f));
  }

  // return errors
  return errors;
}


//*******************************************
//*** disparity error average, kt 2012 ******
//*******************************************
vector<float> disparityErrorsAverage (
		DisparityImage &D_gt,
		DisparityImage &D_orig,
		DisparityImage &D_ipol,
		bool refl) {

  // check file size
  if (D_gt.width()!=D_orig.width() || D_gt.height()!=D_orig.height()) {
    cout << "ERROR: Wrong file size!" << endl;
    throw 1;
  }

  // extract width and height
  int32_t width  = D_gt.width();
  int32_t height = D_gt.height();

  
  // init errors
	// Updated for rmse metric on 2019/08/31
  vector<float> errors;
  for (int32_t i=0; i< 4; i++)
    errors.push_back(0);
  int32_t num_pixels = 0;
  int32_t num_pixels_result = 0;

  // for all pixels do
  for (int32_t u=0; u<width; u++) {
    for (int32_t v=0; v<height; v++) {
      if (D_gt.isValid(u,v)) {
        float d_err = fabs(D_gt.getDisp(u,v)-D_ipol.getDisp(u,v));
        errors[0] += d_err; // for ame error
        errors[2] += d_err * d_err; // for rmse error
        num_pixels++;
        if (D_orig.isValid(u,v)) {
          errors[1] += d_err;
          errors[3] += d_err * d_err;
          num_pixels_result++;
        }
      }
    }
  }
  
  if (refl) {
  
    // push back counts
    errors.push_back((float)num_pixels);
    errors.push_back((float)num_pixels_result);
  
  } else {

    // normalize errors
    errors[0] /= max((float)num_pixels,1.0f);
    errors[1] /= max((float)num_pixels_result,1.0f);
    errors[2] /= max((float)num_pixels,1.0f);
		errors[2] = sqrt(errors[2]);
    errors[3] /= max((float)num_pixels_result,1.0f);
		errors[3] = sqrt(errors[3]);
  }

  // return errors
  return errors;
}


//**************************************************
//*** disparity from pfm to png, kt 2012/2015 ******
//**************************************************
void pfm2uint16PNG(
		string pfm_result_file,// e.g., == "kitti2012-pfm-submit01/"
		string disp_suf, // e.g., == "_post_disp0PKLS"
		string png_result_file,// e.g., == "kitti2012-png-submit01/"
		int imgNum
		){ 
  
  string pfm_result_dir = baseDir + "results/" + pfm_result_file;
  string png_result_dir = baseDir + "results/" + png_result_file;
  PFM pfmIO;
  // for all test files do;
  for (int32_t i=0; i < imgNum; i++){
    // file name;
    char prefix[256];
    sprintf(prefix,"%06d_10",i);
    
		string pfm_image_file = pfm_result_dir + "/" + prefix + disp_suf + ".pfm";
		float * p_disp = pfmIO.read_pfm<float>(pfm_image_file);
		//cout << "Done ! reading pfm disparity: " << prefix + disp_suffix + ".pfm\n";

		// construct disparity image from data
		DisparityImage D_pfm( p_disp, pfmIO.getWidth(), pfmIO.getHeight());
		string png_image_file = png_result_dir + "/" + prefix + ".png";
		D_pfm.write(png_image_file);
		delete[] p_disp;
	}
}


//************************************
//*** disparity from pfm to png ******
//************************************
void pfm2uint16PNG2(
		std::string src_pfm_dir, // input pfm dirs;
		std::string dst_png_dir, // output png dirs;
		boost::python::list img_names	// image lists;
		){
  PFM pfmIO;
  for (int i = 0; i < len(img_names); ++i){
		string img = boost::python::extract<std::string>(img_names[i]);
		string src = src_pfm_dir + "/" + img + ".pfm";
		string dst = dst_png_dir + "/" + img + ".png";
		float * p_disp = pfmIO.read_pfm<float>(src);
		// set the inf value to 0;
		const int w = pfmIO.getWidth();
		const int h = pfmIO.getHeight();
		for (int i = 0; i < w*h; ++i){
			// uint16 : 0 to 65535;
			// The maximum value for uint16 is 65535;
			p_disp[i] = std::isinf(p_disp[i]) ? (uint16_t)65535 : p_disp[i];
		}

		// construct disparity image from data
		DisparityImage D_pfm( p_disp, w, h);
		D_pfm.write(dst);
		delete[] p_disp;
	}
}

inline std::string change_to_path(const std::string & p){
	if (p.back() != '/')
		return p + '/';
	else
		return p;
}

//***********************************
//*** evaluate stereo, kt 2012 ******
//***********************************
bool eval (
		       std::string data_dir,
					 // updated as full dir, e.g.,"/home/ccj/GCNet/results/cbmvnet-gc-regu-sfepo5-F8-testKT12/sfepo5-board50/"
		       std::string result_sha, 
		       std::string evaluate, // e.g., == "rf_evaluate" or "sgm_evaluate"
					 float * err_result, // the accuracy/error result
           // err_result  = {mean_out_noc, min_out_noc, max_out_noc,
           //                mean_avg_noc, min_avg_noc, max_avg_noc}
					 int imgIdxStart, 
					 int imgIdxEnd,
					 int kt2012_2015_type,// if kt-15 wants to use the same error metrics as kt 12.
           Mail* mail // e.g. == NULL
           ){

  std::string gt_noc_dir,gt_occ_dir, gt_refl_noc_dir, gt_refl_occ_dir, gt_img_dir,result_dir, errormap_dir;
  // ground truth and result directories
	if (kt2012_2015_type == 0){
		if (data_dir == string("")){
			imgDir = baseDir + "datasets/KITTI-2012/training/";
		}
		else
			imgDir =change_to_path( data_dir); //i.e., add '/' if possible;
    gt_noc_dir      =  imgDir + "disp_noc";
    gt_occ_dir      =  imgDir + "disp_occ";
		gt_refl_noc_dir =  imgDir + "disp_refl_noc";
    gt_refl_occ_dir =  imgDir + "disp_refl_occ";
    gt_img_dir      =  imgDir + "image_0";
		result_dir      =  result_sha;
		//cout << "result_dir = " << result_dir << ", " << change_to_path(result_dir) << "\n";
		errormap_dir    =  change_to_path(result_dir) + "errorMap-" + evaluate;
		//cout << "Processing " << imgDir << "\n";
	}

	else if (kt2012_2015_type == 1){
		if (data_dir == std::string(""))
			imgDir= baseDir + "datasets/KITTI-2015/training/";
		else
			imgDir= change_to_path(data_dir);
    gt_noc_dir      =  imgDir + "disp_noc_0";
    gt_occ_dir      =  imgDir + "disp_occ_0";
		gt_refl_noc_dir =  imgDir + "disp_refl_noc"; // actually not used, no this directory at all.
    gt_refl_occ_dir =  imgDir + "disp_refl_occ"; // actually not used, no this directory at all.
    gt_img_dir      =  imgDir + "image_0";
		result_dir      =  result_sha;
		errormap_dir    =  change_to_path(result_dir) + "errorMap-" + evaluate;
		//cout << "Processing " << imgDir << "\n";
	}

	else
		cout << "Wrong parameter! Please specify which dataset: kt2012 (kt2012_2015_type = 0) or kt2015 (kt2012_2015_type = 1).\n";

#if isShowErrorMap
			system(("mkdir " + errormap_dir + "/").c_str());
#endif

// check availability of results
   const int num_test_imgs = imgIdxEnd - imgIdxStart;
   bool eval_disp = resultsAvailable(result_sha, imgIdxStart, imgIdxEnd, mail);
	 
	 if (!eval_disp){
		 if (mail)
			 mail->msg("Not enough result images found for stereo evaluation, stopping evaluation.");
		 else
			 printf("Not enough result images found for stereo evaluation, stopping evaluation.");
		 return false;
	 }
	 
	 else {// if eval_disp == true
		 if (mail)
			 mail->msg("Evaluating stereo for %d images.", num_test_imgs);
		 if (isDisplay)
			 printf("Evaluating stereo for %d images.", num_test_imgs);

			// vector for storing the errors
			vector< vector<float> > errors_noc_out;
			vector< vector<float> > errors_occ_out;
			vector< vector<float> > errors_noc_avg;
			vector< vector<float> > errors_occ_avg;
			//vector< vector<float> > errors_refl_noc_out;
			//vector< vector<float> > errors_refl_occ_out;
			//vector< vector<float> > errors_refl_noc_avg;
			//vector< vector<float> > errors_refl_occ_avg;


			PFM pfmIO;
			// for all test files do
			for (int32_t i= (int32_t)imgIdxStart; i< (int32_t)imgIdxEnd; i++) {
				// file name
		    char prefix[256];
				sprintf(prefix,"%06d_10",i);
		
				// catch errors, when loading fails
				try {
					// load ground truth disparity map
					DisparityImage D_gt_noc(gt_noc_dir + "/" + prefix + ".png");
					DisparityImage D_gt_occ(gt_occ_dir + "/" + prefix + ".png");
					//DisparityImage D_gt_refl_noc(gt_refl_noc_dir + "/" + prefix + ".png");
					//DisparityImage D_gt_refl_occ(gt_refl_occ_dir + "/" + prefix + ".png");
					
					// check submitted result
					string pfm_image_file = result_dir + "/" + prefix + disp_suffix + ".pfm";
					float * p_disp = pfmIO.read_pfm<float>(pfm_image_file);
					//cout << "Done ! reading pfm disparity: " << prefix + disp_suffix + ".pfm\n";

					// load submitted result and interpolate missing values
					// construct disparity image from data
					DisparityImage D_orig(p_disp,pfmIO.getWidth(), pfmIO.getHeight());
					DisparityImage D_ipol(D_orig);
					D_ipol.interpolateBackground();
					delete [] p_disp;

					// add disparity errors
					vector<float> errors_noc_out_curr = disparityErrorsOutlier(D_gt_noc,D_orig,D_ipol,false);
					vector<float> errors_occ_out_curr = disparityErrorsOutlier(D_gt_occ,D_orig,D_ipol,false);
					vector<float> errors_noc_avg_curr = disparityErrorsAverage(D_gt_noc,D_orig,D_ipol,false);
					vector<float> errors_occ_avg_curr = disparityErrorsAverage(D_gt_occ,D_orig,D_ipol,false);
					errors_noc_out.push_back(errors_noc_out_curr);
					errors_occ_out.push_back(errors_occ_out_curr);
					errors_noc_avg.push_back(errors_noc_avg_curr);
					errors_occ_avg.push_back(errors_occ_avg_curr);
					//vector<float> errors_refl_noc_out_curr = disparityErrorsOutlier(D_gt_refl_noc,D_orig,D_ipol,true);
					//vector<float> errors_refl_occ_out_curr = disparityErrorsOutlier(D_gt_refl_occ,D_orig,D_ipol,true);
					//vector<float> errors_refl_noc_avg_curr = disparityErrorsAverage(D_gt_refl_noc,D_orig,D_ipol,true);
					//vector<float> errors_refl_occ_avg_curr = disparityErrorsAverage(D_gt_refl_occ,D_orig,D_ipol,true);
					//errors_refl_noc_out.push_back(errors_refl_noc_out_curr);
					//errors_refl_occ_out.push_back(errors_refl_occ_out_curr);
					//errors_refl_noc_avg.push_back(errors_refl_noc_avg_curr);
					//errors_refl_occ_avg.push_back(errors_refl_occ_avg_curr);

				  // save detailed infos for first 20 images
#if isShowErrorMap
			    if (i < imgIdxStart + NUM_ERROR_IMAGES){
						// save errors of error images to text file
						//FILE *errors_noc_out_file = fopen((errormap_dir + "/errors_noc_out.txt").c_str(),"w");
						//FILE *errors_occ_out_file = fopen((errormap_dir + "/errors_occ_out.txt").c_str(),"w");
						//FILE *errors_noc_avg_file = fopen((errormap_dir + "/errors_noc_avg.txt").c_str(),"w");
						//FILE *errors_occ_avg_file = fopen((errormap_dir + "/errors_occ_avg.txt").c_str(),"w");
						
						FILE *errors_noc_file = fopen((errormap_dir + "/errors_noc_out.txt").c_str(),"a");
						FILE *errors_occ_file = fopen((errormap_dir + "/errors_occ_out.txt").c_str(),"a");
						if (errors_noc_file==NULL || errors_occ_file==NULL) {
							if (mail) mail->msg("ERROR: Couldn't generate/store output statistics!");
							else cout << "ERROR: Couldn't generate/store output statistics!\n";
							return false;
						}
            
						fprintf(errors_noc_file,"##****** images : %s\n", prefix);
						fprintf(errors_noc_file,"%f,%f,%f,%f,%f\n", 
								errors_noc_out_curr[thred_3_idx], // 3-thred out, ground truth available regions.
								errors_noc_out_curr[thred_3_idx + 1], //3-thred out, our result available regions.
							 	errors_noc_avg_curr[0], // only 2 elements.
							 	errors_noc_avg_curr[1], // only 2 elements.
							 	errors_noc_out_curr[12] // density.
								);
						fprintf(errors_occ_file,"##****** images : %s\n", prefix);
						fprintf(errors_occ_file,"%f,%f,%f,%f,%f\n", 
								errors_occ_out_curr[thred_3_idx], 
								errors_occ_out_curr[thred_3_idx+1], 
								errors_occ_avg_curr[0],
								errors_occ_avg_curr[1],
								errors_occ_out_curr[12] // density
								);

						//for (int32_t j=0; j<errors_noc_out_curr.size(); j++){
						//	fprintf(errors_noc_out_file,"%f ",errors_noc_out_curr[j]);
						//	fprintf(errors_occ_out_file,"%f ",errors_occ_out_curr[j]);
						//}

						fclose(errors_noc_file);
						fclose(errors_occ_file);

						// save error image
						//png::image<png::rgb_pixel> D_err = D_ipol.errorImage(D_gt_noc,D_gt_occ);
						//D_err.write( errormap_dir + "/" + prefix + ".png");
						
						// save error image
						string imgName = errormap_dir + "/" + prefix + ".jpg";
						D_ipol.errorImage_pkls(D_gt_noc, D_gt_occ, imgName);

						// compute maximum disparity
						//float max_disp = D_gt_occ.maxDisp();
									 
						// save original disparity image false color coded
						//D_orig.writeColor(result_dir + "/"+evaluate+"/disp_orig/" + prefix + ".png",max_disp);
						
						// save interpolated disparity image false color coded
						//D_ipol.writeColor(result_dir + "/"+evaluate+"/disp_ipol/" + prefix + ".png",max_disp);

						// copy left camera image        
						string img_src = gt_img_dir   + "/" + prefix + ".png";
						string img_dst = errormap_dir + "/" + prefix + ".png";
						system(("cp " + img_src + " " + img_dst).c_str());
					} // end of saving detailed infos for first 20 images
#endif
					//on error, exit
				} catch (...) {
					if (mail) mail->msg("ERROR: Couldn't read: %s.png", prefix);
					else cout << "ERROR: Couldn't read: " << prefix << ".png or .pfm\n";
					return false;
		}
	}// end of error evaluation for each image.

#if 0
	// open stats file for writing
	string stats_noc_out_file_name = result_dir + "/"+evaluate+"/stats_noc_out.txt";
	string stats_occ_out_file_name = result_dir + "/"+evaluate+"/stats_occ_out.txt";
	string stats_noc_avg_file_name = result_dir + "/"+evaluate+"/stats_noc_avg.txt";
	string stats_occ_avg_file_name = result_dir + "/"+evaluate+"/stats_occ_avg.txt";
	string stats_refl_noc_out_file_name = result_dir + "/"+evaluate+"/stats_refl_noc_out.txt";
	string stats_refl_occ_out_file_name = result_dir + "/"+evaluate+"/stats_refl_occ_out.txt";
	string stats_refl_noc_avg_file_name = result_dir + "/"+evaluate+"/stats_refl_noc_avg.txt";
	string stats_refl_occ_avg_file_name = result_dir + "/"+evaluate+"/stats_refl_occ_avg.txt";  
	FILE *stats_noc_out_file = fopen(stats_noc_out_file_name.c_str(),"w");
	FILE *stats_occ_out_file = fopen(stats_occ_out_file_name.c_str(),"w");
	FILE *stats_noc_avg_file = fopen(stats_noc_avg_file_name.c_str(),"w");
	FILE *stats_occ_avg_file = fopen(stats_occ_avg_file_name.c_str(),"w");
	FILE *stats_refl_noc_out_file = fopen(stats_refl_noc_out_file_name.c_str(),"w");
	FILE *stats_refl_occ_out_file = fopen(stats_refl_occ_out_file_name.c_str(),"w");
	FILE *stats_refl_noc_avg_file = fopen(stats_refl_noc_avg_file_name.c_str(),"w");
	FILE *stats_refl_occ_avg_file = fopen(stats_refl_occ_avg_file_name.c_str(),"w");
	if (stats_noc_out_file==NULL || stats_occ_out_file==NULL || errors_noc_out.size()==0 || errors_occ_out.size()==0 ||
			stats_noc_avg_file==NULL || stats_occ_avg_file==NULL || errors_noc_avg.size()==0 || errors_occ_avg.size()==0 ||
			stats_refl_noc_out_file==NULL || stats_refl_occ_out_file==NULL || errors_refl_noc_out.size()==0 || errors_refl_occ_out.size()==0 ||
			stats_refl_noc_avg_file==NULL || stats_refl_occ_avg_file==NULL || errors_refl_noc_avg.size()==0 || errors_refl_occ_avg.size()==0) {
		if (mail) mail->msg("ERROR: Couldn't generate/store output statistics!");
		else cout << "ERROR: Couldn't generate/store output statistics!\n";
		return false;
	}

	// write mean
	for (int32_t i=0; i<errors_noc_out[0].size(); i++) {
		fprintf(stats_noc_out_file,"%f ",statMean(errors_noc_out,i));
		fprintf(stats_occ_out_file,"%f ",statMean(errors_occ_out,i));
	}
	for (int32_t i=0; i<errors_noc_avg[0].size(); i++) {
		fprintf(stats_noc_avg_file,"%f ",statMean(errors_noc_avg,i));
		fprintf(stats_occ_avg_file,"%f ",statMean(errors_occ_avg,i));
	}
	fprintf(stats_noc_out_file,"\n");
	fprintf(stats_occ_out_file,"\n");
	fprintf(stats_noc_avg_file,"\n");
	fprintf(stats_occ_avg_file,"\n");

	// write min
	for (int32_t i=0; i<errors_noc_out[0].size(); i++) {
		fprintf(stats_noc_out_file,"%f ",statMin(errors_noc_out,i));
		fprintf(stats_occ_out_file,"%f ",statMin(errors_occ_out,i));
	}
	for (int32_t i=0; i<errors_noc_avg[0].size(); i++) {
		fprintf(stats_noc_avg_file,"%f ",statMin(errors_noc_avg,i));
		fprintf(stats_occ_avg_file,"%f ",statMin(errors_occ_avg,i));
	}
	fprintf(stats_noc_out_file,"\n");
	fprintf(stats_occ_out_file,"\n");
	fprintf(stats_noc_avg_file,"\n");
	fprintf(stats_occ_avg_file,"\n");

	// write max
	for (int32_t i=0; i<errors_noc_out[0].size(); i++) {
		fprintf(stats_noc_out_file,"%f ",statMax(errors_noc_out,i));
		fprintf(stats_occ_out_file,"%f ",statMax(errors_occ_out,i));
	}
	for (int32_t i=0; i<errors_noc_avg[0].size(); i++) {
		fprintf(stats_noc_avg_file,"%f ",statMax(errors_noc_avg,i));
		fprintf(stats_occ_avg_file,"%f ",statMax(errors_occ_avg,i));
	}
	fprintf(stats_noc_out_file,"\n");
	fprintf(stats_occ_out_file,"\n");
	fprintf(stats_noc_avg_file,"\n");
	fprintf(stats_occ_avg_file,"\n");

	// write reflection files
	for (int32_t i=0; i<10; i++) {
		fprintf(stats_refl_noc_out_file,"%f ",statWeightedMean(errors_refl_noc_out,i,10+i%2));
		fprintf(stats_refl_occ_out_file,"%f ",statWeightedMean(errors_refl_occ_out,i,10+i%2));
	}

	for (int32_t i=0; i<2; i++) {
		fprintf(stats_refl_noc_avg_file,"%f ",statWeightedMean(errors_refl_noc_avg,i,2+i%2));
		fprintf(stats_refl_occ_avg_file,"%f ",statWeightedMean(errors_refl_occ_avg,i,2+i%2));
	}

	fprintf(stats_noc_out_file,"\n");
	fprintf(stats_occ_out_file,"\n");
	fprintf(stats_noc_avg_file,"\n");
	fprintf(stats_occ_avg_file,"\n");

	// close files
	fclose(stats_noc_out_file);
	fclose(stats_occ_out_file);
	fclose(stats_noc_avg_file);
	fclose(stats_occ_avg_file);
	fclose(stats_refl_noc_out_file);
	fclose(stats_refl_occ_out_file);
	fclose(stats_refl_noc_avg_file);
	fclose(stats_refl_occ_avg_file);
#endif

	// saving result.
	// err_result  = {mean_out_noc, min_out_noc, max_out_noc,
	//                mean_avg_noc, min_avg_noc, max_avg_noc}
		err_result[0] = statMean(errors_noc_out, thred_3_idx);
		err_result[1] = statMean(errors_occ_out, thred_3_idx);
		err_result[2] = statMean(errors_noc_avg, 0);// only two elements.
		err_result[3] = statMean(errors_occ_avg, 0);// only two elements.
		err_result[4] = statMean(errors_noc_avg, 2);// only two elements.
		err_result[5] = statMean(errors_occ_avg, 2);// only two elements.
	
	//err_result[1] = statMin (errors_noc_out, thred_3_idx);
	//err_result[2] = statMax (errors_noc_out, thred_3_idx);
	//err_result[4] = statMin (errors_noc_avg, 0);
	//err_result[5] = statMax (errors_noc_avg, 0);
	
	 }// if eval_disp == true;
  // success
	return true;
}



//***********************************
//*** evaluate stereo, kt 2012 ******
//***********************************
bool eval_kt2012_from_img_list (
		       std::string data_dir,
					 // updated as full dir, e.g.,"/home/ccj/GCNet/results/cbmvnet-gc-regu-sfepo5-F8-testKT12/sfepo5-board50/"
		       std::string result_sha, 
		       std::string evaluate, // e.g., == "rf_evaluate" or "sgm_evaluate"
					 float * err_result, // the accuracy/error result
           // err_result  = {mean_out_noc, min_out_noc, max_out_noc,
           //                mean_avg_noc, min_avg_noc, max_avg_noc}
					 std::vector<std::string> v_imgs, // image names;
           Mail* mail // e.g. == NULL
           ){

  std::string gt_noc_dir,gt_occ_dir, gt_refl_noc_dir, gt_refl_occ_dir, gt_img_dir,result_dir, errormap_dir;
  // ground truth and result directories
	if (data_dir == string("")){
		imgDir = baseDir + "datasets/KITTI-2012/training/";
	}
	else
		imgDir =change_to_path( data_dir); //i.e., add '/' if possible;
	gt_noc_dir      =  imgDir + "disp_noc";
	gt_occ_dir      =  imgDir + "disp_occ";
	gt_refl_noc_dir =  imgDir + "disp_refl_noc";
	gt_refl_occ_dir =  imgDir + "disp_refl_occ";
	gt_img_dir      =  imgDir + "image_0";
	result_dir      =  result_sha;
	//cout << "result_dir = " << result_dir << ", " << change_to_path(result_dir) << "\n";
	errormap_dir    =  change_to_path(result_dir) + "errorMap-" + evaluate;
	//cout << "Processing " << imgDir << "\n";
	

#if isShowErrorMap
			system(("mkdir " + errormap_dir + "/").c_str());
#endif

// check availability of results
   const int num_test_imgs = v_imgs.size();
   //printf("imgs = %d\n", num_test_imgs);

#if 0
	 for (std::vector<std::string>::iterator it = v_imgs.begin();  
			 it != v_imgs.end(); ++it)
		 std::cout << ' ' << *it;
#endif



// check availability of results
   bool eval_disp = resultsAvailable(result_dir, v_imgs);
	 
	 if (!eval_disp){
		 if (mail)
			 mail->msg("Not enough result images found for stereo evaluation, stopping evaluation.");
		 else
			 printf("Not enough result images found for stereo evaluation, stopping evaluation.");
		 return false;
	 }
	 
	 else {// if eval_disp == true
		 if (mail)
			 mail->msg("Evaluating stereo for %d images.", num_test_imgs);
		 if (isDisplay)
			 printf("Evaluating stereo for %d images.", num_test_imgs);

			// vector for storing the errors
			vector< vector<float> > errors_noc_out;
			vector< vector<float> > errors_occ_out;
			vector< vector<float> > errors_noc_avg;
			vector< vector<float> > errors_occ_avg;

			PFM pfmIO;
			// for all test files do
			for (int32_t i= 0; i< num_test_imgs; i++){
				std::string prefix = v_imgs[i];		
		
				// catch errors, when loading fails
				try {
					// load ground truth disparity map
					DisparityImage D_gt_noc(gt_noc_dir + "/" + prefix + ".png");
					DisparityImage D_gt_occ(gt_occ_dir + "/" + prefix + ".png");
					
					// check submitted result
					string pfm_image_file = result_dir + "/" + prefix + disp_suffix + ".pfm";
					float * p_disp = pfmIO.read_pfm<float>(pfm_image_file);
					//cout << "Done ! reading pfm disparity: " << prefix + disp_suffix + ".pfm\n";

					// load submitted result and interpolate missing values
					// construct disparity image from data
					DisparityImage D_orig(p_disp,pfmIO.getWidth(), pfmIO.getHeight());
					DisparityImage D_ipol(D_orig);
					D_ipol.interpolateBackground();
					delete [] p_disp;

					// add disparity errors
					vector<float> errors_noc_out_curr = disparityErrorsOutlier(D_gt_noc,D_orig,D_ipol,false);
					vector<float> errors_occ_out_curr = disparityErrorsOutlier(D_gt_occ,D_orig,D_ipol,false);
					vector<float> errors_noc_avg_curr = disparityErrorsAverage(D_gt_noc,D_orig,D_ipol,false);
					vector<float> errors_occ_avg_curr = disparityErrorsAverage(D_gt_occ,D_orig,D_ipol,false);
					errors_noc_out.push_back(errors_noc_out_curr);
					errors_occ_out.push_back(errors_occ_out_curr);
					errors_noc_avg.push_back(errors_noc_avg_curr);
					errors_occ_avg.push_back(errors_occ_avg_curr);

				  // save detailed infos for first 20 images
#if isShowErrorMap
			    if (i < NUM_ERROR_IMAGES){
						// save errors of error images to text file
						FILE *errors_noc_file = fopen((errormap_dir + "/errors_noc_out.txt").c_str(),"a");
						FILE *errors_occ_file = fopen((errormap_dir + "/errors_occ_out.txt").c_str(),"a");
						if (errors_noc_file==NULL || errors_occ_file==NULL) {
							if (mail) mail->msg("ERROR: Couldn't generate/store output statistics!");
							else cout << "ERROR: Couldn't generate/store output statistics!\n";
							return false;
						}
            
						fprintf(errors_noc_file,"##****** images : %s\n", prefix);
						fprintf(errors_noc_file,"%f,%f,%f,%f,%f\n", 
								errors_noc_out_curr[thred_3_idx], // 3-thred out, ground truth available regions.
								errors_noc_out_curr[thred_3_idx + 1], //3-thred out, our result available regions.
							 	errors_noc_avg_curr[0], // only 2 elements.
							 	errors_noc_avg_curr[1], // only 2 elements.
							 	errors_noc_out_curr[12] // density.
								);
						fprintf(errors_occ_file,"##****** images : %s\n", prefix);
						fprintf(errors_occ_file,"%f,%f,%f,%f,%f\n", 
								errors_occ_out_curr[thred_3_idx], 
								errors_occ_out_curr[thred_3_idx+1], 
								errors_occ_avg_curr[0],
								errors_occ_avg_curr[1],
								errors_occ_out_curr[12] // density
								);

						//for (int32_t j=0; j<errors_noc_out_curr.size(); j++){
						//	fprintf(errors_noc_out_file,"%f ",errors_noc_out_curr[j]);
						//	fprintf(errors_occ_out_file,"%f ",errors_occ_out_curr[j]);
						//}

						fclose(errors_noc_file);
						fclose(errors_occ_file);

						// save error image
						//png::image<png::rgb_pixel> D_err = D_ipol.errorImage(D_gt_noc,D_gt_occ);
						//D_err.write( errormap_dir + "/" + prefix + ".png");
						
						// save error image
						string imgName = errormap_dir + "/" + prefix + ".jpg";
						D_ipol.errorImage_pkls(D_gt_noc, D_gt_occ, imgName);

						// compute maximum disparity
						//float max_disp = D_gt_occ.maxDisp();
									 
						// save original disparity image false color coded
						//D_orig.writeColor(result_dir + "/"+evaluate+"/disp_orig/" + prefix + ".png",max_disp);
						
						// save interpolated disparity image false color coded
						//D_ipol.writeColor(result_dir + "/"+evaluate+"/disp_ipol/" + prefix + ".png",max_disp);

						// copy left camera image        
						string img_src = gt_img_dir   + "/" + prefix + ".png";
						string img_dst = errormap_dir + "/" + prefix + ".png";
						system(("cp " + img_src + " " + img_dst).c_str());
					} // end of saving detailed infos for first 20 images
#endif
					//on error, exit
				} catch (...) {
					if (mail) mail->msg("ERROR: Couldn't read: %s.png", prefix);
					else cout << "ERROR: Couldn't read: " << prefix << ".png or .pfm\n";
					return false;
		}
	}// end of error evaluation for each image.


	// saving result.
	// err_result  = {mean_out_noc, min_out_noc, max_out_noc,
	//                mean_avg_noc, min_avg_noc, max_avg_noc}
		err_result[0] = statMean(errors_noc_out, thred_3_idx);
		err_result[1] = statMean(errors_occ_out, thred_3_idx);
		err_result[2] = statMean(errors_noc_avg, 0);// only two elements.
		err_result[3] = statMean(errors_occ_avg, 0);// only two elements.
		err_result[4] = statMean(errors_noc_avg, 2);// only two elements.
		err_result[5] = statMean(errors_occ_avg, 2);// only two elements.
	
	
	 }// if eval_disp == true;
  // success
	return true;
}


//**********************************************
//*** check results availability, kt 2015 ******
//**********************************************
inline bool resultsAvailable (
		const string & result_sha, // e.g., "/home/ccj/CBMV-MP/results/kt15-ft-disp-cbmv/"
		const int & imgIdxStart,
		const int & imgIdxEnd,
		Mail * mail
		){
	//string result_dir = baseDir + "results/" + result_sha;
	string result_dir = result_sha;
  int32_t count = 0;
	const int num_test_imgs = imgIdxEnd - imgIdxStart;
  for (int32_t i= imgIdxStart; i< imgIdxEnd; i++) {
    char prefix[256];
    sprintf(prefix,"%06d_10",i);
    FILE *tmp_file = fopen(( change_to_path(result_dir) + prefix + disp_suffix +  ".pfm").c_str(),"rb");
    if (tmp_file) {
      count++;
      fclose(tmp_file);
    }
  }
#if 0
  if (mail)
		mail->msg("Found %d/%d images in %s folder.",count, num_test_imgs, result_sha.c_str());
	else
		printf("Found %d/%d images in %s folder.", count, num_test_imgs, result_sha.c_str());

#endif

	return count== num_test_imgs;
}



//**********************************************
//*** check results availability, kt 2015 ******
//**********************************************
inline bool resultsAvailable(
		const string & result_dir, // e.g., "/home/ccj/CBMV-MP/results/kt15-ft-disp-cbmv/"
		std::vector<std::string> & v_imgs
		){
  int32_t count = 0;
	const int num_test_imgs = v_imgs.size();
  for (int32_t i= 0; i< num_test_imgs; i++) {
    FILE *tmp_file = fopen((result_dir + v_imgs[i] + disp_suffix + ".pfm").c_str(),"rb");
    if (tmp_file) {
      count++;
      fclose(tmp_file);
    }
  }
	return count == num_test_imgs;
}

//***********************************
//*** evaluate stereo, kt 2015 ******
//***********************************
bool eval_kt2015_from_img_list (
		string imgDir, // e.g., == "datasets/KITTI-2015/training/";
		string result_dir, // e.g., == "results/kt15-disp-gcnet/";
		string evaluate, // e.g., == "rf_evaluate" or "post_evaluate"
		const string & timeStamp,
		float * err_result,
		std::vector<std::string> v_imgs, // image names;
		Mail* mail = NULL){

  // ground truth and result directories
  string gt_img_dir = imgDir + "image_0";
  string gt_obj_map_dir = imgDir + "obj_map";
  string gt_disp_noc_0_dir = imgDir + "disp_noc_0";
  string gt_disp_occ_0_dir = imgDir + "disp_occ_0";
  string errormap_dir = result_dir + "errorMap-" + evaluate;
  //std::cout << "gt = " << gt_img_dir << "\n";
	//std::cout << "result_dir = " << result_dir << "\n";
#if isShowErrorMap
		system(("mkdir " + errormap_dir + "/").c_str());
#endif
  // check availability of results
	const int num_test_imgs = v_imgs.size();
  //printf("imgs = %d\n", num_test_imgs);

#if 0
	for (std::vector<std::string>::iterator it = v_imgs.begin();  
			it != v_imgs.end(); ++it)
		    std::cout << ' ' << *it;
#endif

	bool eval_disp = resultsAvailable(result_dir, v_imgs);
  // make sure we have something to evaluate at all
  if (!eval_disp){
		if (mail)
			mail->msg("Not enough result images found for stereo evaluation, stopping evaluation.");
		else
			printf("Not enough result images found for stereo evaluation, stopping evaluation.");
  	return false;
  }
  else {
		if (mail)
			mail->msg("Evaluating stereo for %d images.", num_test_imgs);
		if (isDisplay)
			printf("Evaluating stereo for %d images.", num_test_imgs);
	}

  // accumulators
  float errors_disp_noc_0[3*4+6]  = {0,0,0,0,0,0,0,0,0,0,0,0, 
		                                  0,0,0,0,0,0 //added for mae and rmse errors
	                                };
  float errors_disp_occ_0[3*4+6]  = {0,0,0,0,0,0,0,0,0,0,0,0, 
		                               0,0,0,0,0,0};
  
	PFM pfmIO;
  // for all test files do
  for (int32_t i= 0; i< num_test_imgs; i++){
		std::string prefix = v_imgs[i];
    //cout << "prefix = " << prefix << "\n";
		
		// catch errors, when loading fails
    try {
      // declaration of global data structures
      DisparityImage D_gt_noc_0, D_gt_occ_0;
      // load object map (0:background, >0:foreground)
      IntegerImage O_map = IntegerImage(gt_obj_map_dir + "/" + prefix + ".png");
 
#if isShowErrorMap
      // copy left camera image 
      if (i < NUM_ERROR_IMAGES) {       
        string img_src = gt_img_dir   + "/" + prefix + ".png";
        string img_dst = errormap_dir + "/" + prefix + ".png";
        system(("cp " + img_src + " " + img_dst).c_str());
      }
#endif
      /////////////////////////////////////////////
      // evaluation of disp 0
      if (eval_disp) {
        
        // load ground truth disparity maps
        D_gt_noc_0 = DisparityImage(gt_disp_noc_0_dir + "/" + prefix + ".png");
        D_gt_occ_0 = DisparityImage(gt_disp_occ_0_dir + "/" + prefix + ".png");

        // check submitted result
				string pfm_image_file = result_dir + prefix + disp_suffix + ".pfm";
				float * p_disp = pfmIO.read_pfm<float>(pfm_image_file);
				//cout << "pfm_image_file : " << pfm_image_file << "\n";
				if (isDisplay)
					cout << "Done ! reading pfm disparity: " << prefix + disp_suffix + ".pfm\n";


        // load submitted result and interpolate missing values
				DisparityImage D_orig_0(p_disp, pfmIO.getWidth(), pfmIO.getHeight());
        DisparityImage D_ipol_0 = DisparityImage(D_orig_0);
        D_ipol_0.interpolateBackground();
				delete [] p_disp;
        
				bool is_mae_rmse = true;
        // calculate disparity errors
				// cout << "line 1150\n";
        vector<float> errors_noc_curr = disparityErrorsOutlier_kt15(
						D_gt_noc_0, 
						D_orig_0, 
						D_ipol_0, 
						O_map, 
						is_mae_rmse);

        vector<float> errors_occ_curr = disparityErrorsOutlier_kt15(
						D_gt_occ_0, 
						D_orig_0, 
						D_ipol_0, 
						O_map, 
						is_mae_rmse);

        // accumulate errors
        for (int32_t j=0; j < errors_noc_curr.size()-1; j++) {
					//cout << "index j = " << j << std::endl;
          errors_disp_noc_0[j] += errors_noc_curr[j];
          errors_disp_occ_0[j] += errors_occ_curr[j];
        }
#if isShowErrorMasp
        // save error images
        if (i< NUM_ERROR_IMAGES) {

          // save errors of error images to text file
          FILE *errors_noc_file = fopen((errormap_dir + "/errors_disp_noc_0.txt").c_str(),"a");
          FILE *errors_occ_file = fopen((errormap_dir + "/errors_disp_occ_0.txt").c_str(),"a"); 
          
					fprintf(errors_noc_file,"##****** images : %s\n", prefix);
					for (int32_t i=0; i<12; i+=2)
            fprintf(errors_noc_file,"%f,",errors_noc_curr[i]/max(errors_noc_curr[i+1],1.0f));
          fprintf(errors_noc_file,"%f\n",errors_noc_curr[12]);
          
					fprintf(errors_occ_file,"##****** images : %s\n", prefix);
          for (int32_t i=0; i<12; i+=2)
            fprintf(errors_occ_file,"%f,",errors_occ_curr[i]/max(errors_occ_curr[i+1],1.0f));
          fprintf(errors_occ_file,"%f\n",errors_occ_curr[12]);
          
					fclose(errors_noc_file);
          fclose(errors_occ_file);

          // save error image
					string imgName = errormap_dir + "/" + prefix + ".jpg";
          D_ipol_0.errorImage_pkls(D_gt_noc_0,D_gt_occ_0,imgName);

        }
#endif
      }// end of evaluation for 1 image;


    // on error, exit
    } 
		catch (...) {
			if (mail)
				mail->msg("ERROR: Couldn't read: %s.png", prefix);
			else
				printf("ERROR: Couldn't read: %s.png", prefix);
      return false;
    }
	}// end of evaluation for all the images;


  string stats_file_name;
  FILE *stats_file;


	// saving result.
	char * notes[] = {
		"noc-bg  (all       pixels)",
		"noc-bg  (estimated pixels)",
		"noc-fg  (all       pixels)",
		"noc-fg  (estimated pixels)",
		"noc-all (all       pixels)",
		"noc-all (estimated pixels)",
		"noc-all (         density)",
		
		"occ-bg  (all       pixels)",
		"occ-bg  (estimated pixels)",
		"occ-fg  (all       pixels)",
		"occ-fg  (estimated pixels)",
		"occ-all (all       pixels)",
		"occ-all (estimated pixels)",
		"occ-all (         density)",

    // added for mae and rmse error;
		"noc-bg  (mae         errs)",
		"noc-fg  (mae         errs)",
		"noc-all (mae         errs)",
		"noc-bg  (rmse        errs)",
		"noc-fg  (rmse        errs)",
		"noc-all (rmse        errs)",

		"occ-bg  (mae         errs)",
		"occ-fg  (mae         errs)",
		"occ-all (mae         errs)",
		"occ-bg  (rmse        errs)",
		"occ-fg  (rmse        errs)",
		"occ-all (rmse        errs)",
	};

	err_result[0] = errors_disp_noc_0[0]/max(errors_disp_noc_0[1],1.0f); // noc, bg (all       pixels)
	err_result[1] = errors_disp_noc_0[2]/max(errors_disp_noc_0[3],1.0f); // noc, bg (estimated pixels)
	err_result[2] = errors_disp_noc_0[4]/max(errors_disp_noc_0[5],1.0f); // noc, fg (all       pixels)
	err_result[3] = errors_disp_noc_0[6]/max(errors_disp_noc_0[7],1.0f); // noc, fg (estimated pixels)
	err_result[4] = errors_disp_noc_0[8]/max(errors_disp_noc_0[9],1.0f); // noc,all (all       pixels)
	err_result[5] = errors_disp_noc_0[10]/max(errors_disp_noc_0[11],1.0f);//noc,all (estimated pixels)
	err_result[6] = errors_disp_noc_0[11]/max(errors_disp_noc_0[9],1.0f); //noc,all (density)
	
	err_result[7] = errors_disp_occ_0[0]/ max(errors_disp_occ_0[1],1.0f); // occ, bg (all       pixels)
	err_result[8] = errors_disp_occ_0[2]/ max(errors_disp_occ_0[3],1.0f); // occ, bg (estimated pixels)
	err_result[9] = errors_disp_occ_0[4]/ max(errors_disp_occ_0[5],1.0f); // occ, fg (all       pixels)
	err_result[10]= errors_disp_occ_0[6]/ max(errors_disp_occ_0[7],1.0f); // occ, fg (estimated pixels)
	err_result[11]= errors_disp_occ_0[8]/ max(errors_disp_occ_0[9],1.0f); // occ,all (all       pixels)
	err_result[12]= errors_disp_occ_0[10]/max(errors_disp_occ_0[11],1.0f);//occ,all (estimated pixels)
	err_result[13]= errors_disp_occ_0[11]/max(errors_disp_occ_0[9],1.0f); //occ,all (density)
	
  // added for mae and rmse error;
	err_result[14]= errors_disp_noc_0[12]/ max(errors_disp_noc_0[1],1.0f);// noc, bg (mae errs)
	err_result[15]= errors_disp_noc_0[13]/ max(errors_disp_noc_0[5],1.0f);// noc, fg (mae errs)
	err_result[16]= errors_disp_noc_0[14]/ max(errors_disp_noc_0[9],1.0f);// noc, all (mae errs)
	err_result[17]= sqrt(errors_disp_noc_0[15]/ max(errors_disp_noc_0[1],1.0f));// noc, bg (rmse errs)
	err_result[18]= sqrt(errors_disp_noc_0[16]/ max(errors_disp_noc_0[5],1.0f));// noc, fg (rmse errs)
	err_result[19]= sqrt(errors_disp_noc_0[17]/ max(errors_disp_noc_0[9],1.0f));// noc, all (rmse errs)

	
	err_result[20]= errors_disp_occ_0[12]/ max(errors_disp_occ_0[1],1.0f);// occ, bg (mae errs)
	err_result[21]= errors_disp_occ_0[13]/ max(errors_disp_occ_0[5],1.0f);// occ, fg (mae errs)
	err_result[22]= errors_disp_occ_0[14]/ max(errors_disp_occ_0[9],1.0f);// occ, all (mae errs)
	err_result[23]= sqrt(errors_disp_occ_0[15]/ max(errors_disp_occ_0[1],1.0f));// occ, bg (rmse errs)
	err_result[24]= sqrt(errors_disp_occ_0[16]/ max(errors_disp_occ_0[5],1.0f));// occ, fg (rmse errs)
	err_result[25]= sqrt(errors_disp_occ_0[17]/ max(errors_disp_occ_0[9],1.0f));// occ, all (rmse errs)
	
	//cout << "line 1285\n";
	
  // write summary statistics for disparity evaluation
  if (eval_disp) {
    stats_file_name = result_dir + "/stats_disp_0.txt";
    stats_file = fopen(stats_file_name.c_str(),"a");
    fprintf(stats_file,"##*****%s:\n", timeStamp.c_str());
	  //cout << "line 1292\n";
    for (int32_t i=0; i< 14+12; i++)
      fprintf(stats_file,"%s: %f;\n", notes[i], err_result[i]);
    
		fprintf(stats_file,"\n");
    fclose(stats_file);
  }
  // success
	printf("Successfully evaluated images!!\n");
	return true;
}

//***********************************
//*** evaluate stereo, kt 2015 ******
//***********************************
bool eval_kt2015 (
		string data_dir,
		string result_sha, // e.g., == "KITTI2012-L50K-Cen11Ncc5Sad5Sob5/1p-vs-2n-w-intrp"
		string evaluate, // e.g., == "rf_evaluate" or "post_evaluate"
		const string & timeStamp,
		float * err_result, 
		int imgIdxStart,
		int imgIdxEnd,
		Mail* mail = NULL){
  if (data_dir == std::string(""))
		imgDir = baseDir + "datasets/KITTI-2015/training/";
	else
		imgDir = change_to_path(data_dir);
	
  // ground truth and result directories
  string gt_img_dir = imgDir + "image_0";
  string gt_obj_map_dir = imgDir + "obj_map";
  string gt_disp_noc_0_dir = imgDir + "disp_noc_0";
  string gt_disp_occ_0_dir = imgDir + "disp_occ_0";
  string result_dir = baseDir + "results/" + result_sha;
  string errormap_dir = result_dir + "/" + "errorMap-" + evaluate;

#if 0
  cout << "result_dir = " << result_sha << "\n"
		   << "evaluate_disparity_map_type = " << evaluate << "\n"
			 << "timeStamp = " << timeStamp << "\n"
       << "image number = " << imgIdxEnd - imgIdxStart  << "\n";

#endif

#if isShowErrorMap
		system(("mkdir " + errormap_dir + "/").c_str());
#endif
  // check availability of results
	const int num_test_imgs = imgIdxEnd - imgIdxStart;
  //bool eval_disp = resultsAvailable(result_sha, evaluate, imgIdxStart, imgIdxEnd, mail);
  bool eval_disp = resultsAvailable(result_sha, imgIdxStart, imgIdxEnd, mail);
  // make sure we have something to evaluate at all
  if (!eval_disp){
		if (mail)
			mail->msg("Not enough result images found for stereo evaluation, stopping evaluation.");
		else
			printf("Not enough result images found for stereo evaluation, stopping evaluation.");
  	return false;
  }
  else {
		if (mail)
			mail->msg("Evaluating stereo for %d images.", num_test_imgs);
		if (isDisplay)
			printf("Evaluating stereo for %d images.", num_test_imgs);
	}

  // accumulators
  float errors_disp_noc_0[3*4]     = {0,0,0,0,0,0,0,0,0,0,0,0};
  float errors_disp_occ_0[3*4]     = {0,0,0,0,0,0,0,0,0,0,0,0};
  
	PFM pfmIO;
  // for all test files do
  for (int32_t i= imgIdxStart; i< imgIdxEnd; i++) {

    // file name
    char prefix[256];
    sprintf(prefix,"%06d_10",i);

#if 0  
    // output
		if (mail)
			mail->msg("Processing: %s.png",prefix);
		if (isDisplay)
			printf ("Processing: %s.png", prefix);
#endif

		// catch errors, when loading fails
    try {

      // declaration of global data structures
      DisparityImage D_gt_noc_0, D_gt_occ_0;
      // load object map (0:background, >0:foreground)
      IntegerImage O_map = IntegerImage(gt_obj_map_dir + "/" + prefix + ".png");
 
#if isShowErrorMap
      // copy left camera image 
      if (i < imgIdxStart + NUM_ERROR_IMAGES) {       
        string img_src = gt_img_dir   + "/" + prefix + ".png";
        string img_dst = errormap_dir + "/" + prefix + ".png";
        system(("cp " + img_src + " " + img_dst).c_str());
      }
			
#endif
      ///////////////////////////////////////////////////////////////////////////////////////////
      // evaluation of disp 0
      if (eval_disp) {
        
        // load ground truth disparity maps
        D_gt_noc_0 = DisparityImage(gt_disp_noc_0_dir + "/" + prefix + ".png");
        D_gt_occ_0 = DisparityImage(gt_disp_occ_0_dir + "/" + prefix + ".png");

        // check submitted result
				string pfm_image_file = result_dir + "/" + prefix + disp_suffix + ".pfm";
				float * p_disp = pfmIO.read_pfm<float>(pfm_image_file);
				if (isDisplay)
					cout << "Done ! reading pfm disparity: " << prefix + disp_suffix + ".pfm\n";


        // load submitted result and interpolate missing values
				DisparityImage D_orig_0(p_disp, pfmIO.getWidth(), pfmIO.getHeight());
        DisparityImage D_ipol_0 = DisparityImage(D_orig_0);
        D_ipol_0.interpolateBackground();
				delete [] p_disp;

        // calculate disparity errors
        vector<float> errors_noc_curr = disparityErrorsOutlier_kt15(D_gt_noc_0, D_orig_0, D_ipol_0, O_map);
        vector<float> errors_occ_curr = disparityErrorsOutlier_kt15(D_gt_occ_0, D_orig_0, D_ipol_0, O_map);

        // accumulate errors
        for (int32_t j=0; j < errors_noc_curr.size()-1; j++) {
          errors_disp_noc_0[j] += errors_noc_curr[j];
          errors_disp_occ_0[j] += errors_occ_curr[j];
        }
#if isShowErrorMasp
        // save error images
        if (i< imgIdxStart + NUM_ERROR_IMAGES) {

          // save errors of error images to text file
          FILE *errors_noc_file = fopen((errormap_dir + "/errors_disp_noc_0.txt").c_str(),"a");
          FILE *errors_occ_file = fopen((errormap_dir + "/errors_disp_occ_0.txt").c_str(),"a"); 
          
					fprintf(errors_noc_file,"##****** images : %s\n", prefix);
					for (int32_t i=0; i<12; i+=2)
            fprintf(errors_noc_file,"%f,",errors_noc_curr[i]/max(errors_noc_curr[i+1],1.0f));
          fprintf(errors_noc_file,"%f\n",errors_noc_curr[12]);
          
					fprintf(errors_occ_file,"##****** images : %s\n", prefix);
          for (int32_t i=0; i<12; i+=2)
            fprintf(errors_occ_file,"%f,",errors_occ_curr[i]/max(errors_occ_curr[i+1],1.0f));
          fprintf(errors_occ_file,"%f\n",errors_occ_curr[12]);
          
					fclose(errors_noc_file);
          fclose(errors_occ_file);

          // save error image
					string imgName = errormap_dir + "/" + prefix + ".jpg";
          D_ipol_0.errorImage_pkls(D_gt_noc_0,D_gt_occ_0,imgName);

         //D_ipol_0.errorImage(D_gt_noc_0,D_gt_occ_0,true).write( errormap_dir + "/" + prefix + ".png");

          // compute maximum disparity
          //float max_disp = D_gt_occ_0.maxDisp();

          // save interpolated disparity image false color coded
          //D_ipol_0.writeColor(errormap_dir + "/" + prefix + ".png", max_disp);
        }
#endif
      }// end of evaluation for 1 image;


    // on error, exit
    } 
		catch (...) {
			if (mail)
				mail->msg("ERROR: Couldn't read: %s.png",prefix);
			else
				printf("ERROR: Couldn't read: %s.png",prefix);
      return false;
    }
	}// end of evaluation for all the images;


  string stats_file_name;
  FILE *stats_file;


	// saving result.
	char * notes[] = {
		"noc-bg  (all       pixels)",
		"noc-bg  (estimated pixels)",
		"noc-fg  (all       pixels)",
		"noc-fg  (estimated pixels)",
		"noc-all (all       pixels)",
		"noc-all (estimated pixels)",
		"noc-all (         density)",
		
		"occ-bg  (all       pixels)",
		"occ-bg  (estimated pixels)",
		"occ-fg  (all       pixels)",
		"occ-fg  (estimated pixels)",
		"occ-all (all       pixels)",
		"occ-all (estimated pixels)",
		"ooc-all (         density)"
	};

	err_result[0] = errors_disp_noc_0[0]/max(errors_disp_noc_0[1],1.0f); // noc, bg (all       pixels)
	err_result[1] = errors_disp_noc_0[2]/max(errors_disp_noc_0[3],1.0f); // noc, bg (estimated pixels)
	err_result[2] = errors_disp_noc_0[4]/max(errors_disp_noc_0[5],1.0f); // noc, fg (all       pixels)
	err_result[3] = errors_disp_noc_0[6]/max(errors_disp_noc_0[7],1.0f); // noc, fg (estimated pixels)
	err_result[4] = errors_disp_noc_0[8]/max(errors_disp_noc_0[9],1.0f); // noc,all (all       pixels)
	err_result[5] = errors_disp_noc_0[10]/max(errors_disp_noc_0[11],1.0f);//noc,all (estimated pixels)
	err_result[6] = errors_disp_noc_0[11]/max(errors_disp_noc_0[9],1.0f); //noc,all (density)
	
	err_result[7] = errors_disp_occ_0[0]/ max(errors_disp_occ_0[1],1.0f); // occ, bg (all       pixels)
	err_result[8] = errors_disp_occ_0[2]/ max(errors_disp_occ_0[3],1.0f); // occ, bg (estimated pixels)
	err_result[9] = errors_disp_occ_0[4]/ max(errors_disp_occ_0[5],1.0f); // occ, fg (all       pixels)
	err_result[10]= errors_disp_occ_0[6]/ max(errors_disp_occ_0[7],1.0f); // occ, fg (estimated pixels)
	err_result[11]= errors_disp_occ_0[8]/ max(errors_disp_occ_0[9],1.0f); // occ,all (all       pixels)
	err_result[12]= errors_disp_occ_0[10]/max(errors_disp_occ_0[11],1.0f);//occ,all (estimated pixels)
	err_result[13]= errors_disp_occ_0[11]/max(errors_disp_occ_0[9],1.0f); //occ,all (density)
	
  // write summary statistics for disparity evaluation
  if (eval_disp) {
    stats_file_name = result_dir + "/stats_disp_0.txt";
    stats_file = fopen(stats_file_name.c_str(),"a");
    fprintf(stats_file,"##*****%s:\n", timeStamp.c_str());
    for (int32_t i=0; i<14; i++)
      fprintf(stats_file,"%s: %f;\n", notes[i], err_result[i]);
    
		fprintf(stats_file,"\n");
    fclose(stats_file);
  }
  // success
	// printf("Successfully evaluated %d images!!\n", imgIdxEnd - imgIdxStart);
	return true;
}


PyObject*  evaluate_training_kt2015_from_img_list(
		const string & imgDir, // e.g., == "datasets/KITTI-2015/training/";
	  const string & result_dir, // e.g., == "results/kt15-disp-gcnet/";  
		const string & evaluate_disparity_map_type, 
		const string & disp_suf, // e.g., == "_sgm_disp0PKLS";
		const string & timeStamp,
		boost::python::list img_list // image names without file extension;
		){
	
	// global variable
	disp_suffix = disp_suf;

  // run evaluation
	npy_intp * dims = new npy_intp[1];
	// updated for mae ans rmse errors
  dims[0] = 14 + 12;
	PyObject * errs_A = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
	
	//cout << "new PyObject errs_A\n";
	float * err_result = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(errs_A)));
  
	// updated for mae ans rmse errors
	for (int i = 0; i < dims[0]; i++)
		err_result[i] = -1.0; 

	std::vector<std::string> v_imgs;// image names;
	const int N = len(img_list);
  //printf("N = %d\n", N);
  for (int i = 0; i < N; i++){
		std::string tmp = boost::python::extract<std::string>(img_list[i]);
		v_imgs.push_back(tmp);
	}
	bool success = eval_kt2015_from_img_list(imgDir, result_dir, 
			evaluate_disparity_map_type, timeStamp, err_result, v_imgs, NULL);
  
	if (!success){
		cout << "\n******************************\n"
		     << evaluate_disparity_map_type + " Evaluation failed!\n"
				 <<   "******************************\n";
  }
	return errs_A;
}


PyObject*  evaluate_training_kt2015(
		const string & data_dir, 
	  const string & result_sha, 
		const string & evaluate_disparity_map_type, 
		const string & disp_suf, // e.g., == "_sgm_disp0PKLS";
		const string & timeStamp,
		const int & imgIdxStart,
		const int & imgIdxEnd) {
	
	// global variable
	disp_suffix = disp_suf;

#if 0
  cout << "result_dir = " << result_sha << "\n"
		   << "evaluate_disparity_map_type = " << evaluate_disparity_map_type << "\n"
			 << "disp_suffix = " << disp_suffix << "\n"
			 << "timeStamp = " << timeStamp << "\n"
       << "image number = " << imgIdxEnd - imgIdxStart  << "\n";
#endif
	  
  // run evaluation
	npy_intp * dims = new npy_intp[1];
  dims[0] = 14;
	PyObject * errs_A = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
	//cout << "new PyObject errs_A\n";
	float * err_result = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(errs_A)));
  
	for (int i = 0; i < 14; i++)
		err_result[i] = -1.0;

  //for (int i = 0; i < 14; i++)
	//	cout << err_result[i] << "\n";
  
	//float err_result[14]; // the accuracy/error result
  
	bool success = eval_kt2015(data_dir, result_sha,  evaluate_disparity_map_type, timeStamp, err_result, imgIdxStart, imgIdxEnd, NULL);
  
	//for (int i = 0; i < 14; i++)
	//	cout << err_result[i] << "\n";
  
	if (!success){
		cout << "\n******************************\n"
		     << evaluate_disparity_map_type + " Evaluation failed!\n"
				 <<   "******************************\n";
  }
	return errs_A;
}

PyObject * evaluate_training(
		const string & data_dir,
		const string & result_sha, 
		const string & evaluate_disparity_map_type, 
		const string & disp_suf, // e.g., == "_sgm_disp0PKLS";
    //const string & paramName,
		const string & timeStamp,
    //const float & paramValue,
		const int & imgIdxStart,
		const int & imgIdxEnd,
		const int & kt2012_2015_type) {
	
	// global variable
	disp_suffix = disp_suf;

#if 0
  cout << "result_dir = " << result_sha << "\n"
		   << "evaluate_disparity_map_type = " << evaluate_disparity_map_type << "\n"
			 << "disp_suffix = " << disp_suffix << "\n"
       << "image number = " << imgIdxEnd - imgIdxStart  << "\n";
       //<< "paramName = " << paramName << "\n"
       //<< "timeStamp = " << timeStamp << "\n"
       //<< "paramValue = " << paramValue << "\n";
#endif

	  
  // run evaluation
	int nd = 1;
	npy_intp * dims = new npy_intp[1];
  //dims[0] = 4;
  dims[0] = 4 + 2; // updated for rmse and mae metrc on 2019/08/31
	PyObject* errs_A = PyArray_SimpleNew(nd, dims, NPY_FLOAT32);
	float * err_result = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(errs_A)));	
  //cout << "\n";
	//for (int i = 0; i < dims[0]; i++)
	//	cout << err_result[i] << "\n"; 
	//float err_result[6]; // the accuracy/error result
  bool success = eval(data_dir, result_sha, evaluate_disparity_map_type, err_result, imgIdxStart, imgIdxEnd, kt2012_2015_type,  NULL);


  if (success){
    #if 0
	  //std::cout << "Success!\n";
    string fileName  =  baseDir + "results/" + result_sha + "/" + "disp_err_noc.txt";
    std::ofstream ofs(fileName.c_str(), std::fstream::out| std::fstream::app);
    if (!ofs.is_open()){
      std::cout << "Error! Cannot open file " << fileName << "\n";
    }
    else{
      ofs << timeStamp << ", errors : " << err_result[0] 
          << ","  << err_result[1]
          << ","  << err_result[2]
          << ","  << err_result[3]
          << ","  << err_result[4]
          << ","  << err_result[5]
          << "\n";
      ofs.close();
    }
    #endif
    
    #if 0 
		cout << "\n******************************\n"
		     << evaluate_disparity_map_type + " Evaluation successed!\n"
				 <<   "******************************\n";
    printf("Out-Noc: mean = %f, min = %f, max = %f\n", err_result[0], err_result[1], err_result[2]);
    printf("Avg-Noc: mean = %f, min = %f, max = %f\n", err_result[3], err_result[4], err_result[5]);
    #endif
    //return err_result[0];
    return errs_A;
  }
  else{ 
		cout << "\n******************************\n"
		     << evaluate_disparity_map_type + " Evaluation failed!\n"
				 <<   "******************************\n";
		/*
		err_result[0] = -1.0f;
		err_result[1] = -1.0f;
		err_result[2] = -1.0f;
		err_result[3] = -1.0f;
		err_result[4] = -1.0f;
		err_result[5] = -1.0f;

    return errs_A;
		*/
		//return -1.0;
    return errs_A;
  }
}

//NOTE: newly added on 2020/04/20, by CCJ:
/* for KT2012 evaluation: to evaluate a list of images*/
PyObject * evaluate_training_from_img_list(
		const string & data_dir,
		const string & result_sha, 
		const string & evaluate_disparity_map_type, 
		const string & disp_suf, // e.g., == "_sgm_disp0PKLS";
    //const string & paramName,
		const string & timeStamp,
		boost::python::list img_list // image names without file extension;
		) {
	
	// global variable
	disp_suffix = disp_suf;

#if 0
  cout << "result_dir = " << result_sha << "\n"
		   << "evaluate_disparity_map_type = " << evaluate_disparity_map_type << "\n"
			 << "disp_suffix = " << disp_suffix << "\n"
       << "image number = " << imgIdxEnd - imgIdxStart  << "\n";
       //<< "paramName = " << paramName << "\n"
       //<< "timeStamp = " << timeStamp << "\n"
       //<< "paramValue = " << paramValue << "\n";
#endif

	  
  // run evaluation
	int nd = 1;
	npy_intp * dims = new npy_intp[1];
  //dims[0] = 4;
  dims[0] = 4 + 2; // updated for rmse and mae metrc on 2019/08/31
	PyObject* errs_A = PyArray_SimpleNew(nd, dims, NPY_FLOAT32);
	float * err_result = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(errs_A)));	
  //cout << "\n";
	//for (int i = 0; i < dims[0]; i++)
	//	cout << err_result[i] << "\n"; 
	//float err_result[6]; // the accuracy/error result
	
	std::vector<std::string> v_imgs;// image names;
	const int N = len(img_list);
  //printf("N = %d\n", N);
	for (int i = 0; i < N; i++){
		std::string tmp = boost::python::extract<std::string>(img_list[i]);
		v_imgs.push_back(tmp);
	}

  bool success = eval_kt2012_from_img_list(data_dir, result_sha, evaluate_disparity_map_type, err_result,  v_imgs, NULL);
	


  if (success){
    #if 0
	  //std::cout << "Success!\n";
    string fileName  =  baseDir + "results/" + result_sha + "/" + "disp_err_noc.txt";
    std::ofstream ofs(fileName.c_str(), std::fstream::out| std::fstream::app);
    if (!ofs.is_open()){
      std::cout << "Error! Cannot open file " << fileName << "\n";
    }
    else{
      ofs << timeStamp << ", errors : " << err_result[0] 
          << ","  << err_result[1]
          << ","  << err_result[2]
          << ","  << err_result[3]
          << ","  << err_result[4]
          << ","  << err_result[5]
          << "\n";
      ofs.close();
    }
    #endif
    
    #if 0 
		cout << "\n******************************\n"
		     << evaluate_disparity_map_type + " Evaluation successed!\n"
				 <<   "******************************\n";
    printf("Out-Noc: mean = %f, min = %f, max = %f\n", err_result[0], err_result[1], err_result[2]);
    printf("Avg-Noc: mean = %f, min = %f, max = %f\n", err_result[3], err_result[4], err_result[5]);
    #endif
    //return err_result[0];
    return errs_A;
  }
  else{ 
		cout << "\n******************************\n"
		     << evaluate_disparity_map_type + " Evaluation failed!\n"
				 <<   "******************************\n";
		/*
		err_result[0] = -1.0f;
		err_result[1] = -1.0f;
		err_result[2] = -1.0f;
		err_result[3] = -1.0f;
		err_result[4] = -1.0f;
		err_result[5] = -1.0f;

    return errs_A;
		*/
		//return -1.0;
    return errs_A;
  }
}


BOOST_PYTHON_MODULE(libevaluate_stereo_training){
	/* for Boost <= version 1.63*/
	  
    //numeric::array::set_module_and_type("numpy", "ndarray");
	  /* for Boost > version 1.63*/
		//np::initialize();
//#if BOOST_VERSION >= 106300 // >= 1.63.0
//		np::initialize();
//#else
//    np::array::set_module_and_type("numpy", "ndarray");
//#endif
	np::initialize();
	def("evaluate_training", evaluate_training);
	def("evaluate_training_from_img_list", evaluate_training_from_img_list);
  def("evaluate_training_kt2015", evaluate_training_kt2015);
  def("evaluate_training_kt2015_from_img_list", evaluate_training_kt2015_from_img_list);
	def("pfm2uint16PNG", pfm2uint16PNG);
	def("pfm2uint16PNG2", pfm2uint16PNG2);
	/* 
	 * Error: return-statement with a value, in function returning 'void' [-fpermissive]
	 *        #define NUMPY_IMPORT_ARRAY_RETVAL NULL
	 * > See solution: https://github.com/numpy/numpy/issues/10486
	 * > 1) Solution1: Okay, so the issue occurs only on py2 + py3c (the initialization function is nonstandard, and has py3 semantics). Solution appears to be to use `import_array1()`;
	 * > 2) Solution2: Or call _import_array(), which allows you more control than just `return`;
	 */ 
  //import_array(); // work well for Python2.7;
	import_array1(); // work well for python3.7;
	//_import_array(); // work well for python3.7;
}

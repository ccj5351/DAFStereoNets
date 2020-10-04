// ************************
// ************************
// Edited by CCJ:
// evaldisp_lib.cpp, 
// which can be called
// by python code;
// ************************
// ************************

// evaluate disparity map
// simple version for SDK
// supports upsampling of disp map if GT has higher resolution

// DS 7/2/2014
// 10/14/2014 changed computation of average error
// 1/27/2015 added clipping of valid (non-INF) disparities to [0 .. maxdisp]
//    in fairness to those methods that do not utilize the given disparity range
//    (maxdisp is specified at disp resolution, NOT GT resolution)

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include "imageLib/imageLib.h"
#include <stdint.h> /*for int32_t*/

//added by CCJ for Python + CPP coding;
#include <boost/python.hpp>
#include "boost/python/extract.hpp"
//#include "boost/python/numeric.hpp"
/* Updated by CCJ for Boost version 1.65:
 * Boost 1.65 removes boost/python/numeric.hpp 
 */
// BOOST_LIB_VERSION and BOOST_VERSION are defined in this header file;
#include <boost/version.hpp> 
#if BOOST_VERSION >= 106300 // >= 1.63.0
   #include "boost/python/numpy.hpp"
   namespace np = boost::python::numpy;
#else
   #include "boost/python/numeric.hpp"
	 namespace np = boost::python::numeric;
#endif

#include <numpy/ndarrayobject.h>
using namespace boost::python;

// see Fastest way to check if a file exist using standard C++/C++11/C?
// at https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c;
#include <sys/stat.h>
inline bool exists_file_test(const std::string & name){
	struct stat buffer;
	return (stat (name.c_str(), &buffer) == 0);
}



const int verbose = 0;

/* The `occlusion mask` for the left image is given as a file "mask0nocc.png":
 * - Pixels without ground truth have the color (0, 0, 0).                                                
 * - Pixels which are only observed by the left image have the color (128, 128, 128).
 * - Pixels which are observed by both images have the color (255, 255, 255).       
 * - For the "non-occluded" evaluation, the evaluation is limited to the pixels observed by both images.		
*/


void evaldisp(
		CFloatImage disp, 
		CFloatImage gtdisp, 
		CByteImage mask, 
		float badthresh, 
		int maxdisp, 
		int rounddisp,
		float * statistics
		){

    CShape sh = gtdisp.Shape();
    CShape sh2 = disp.Shape();
    CShape msh = mask.Shape();

    int width = sh.width, height = sh.height;
    int width2 = sh2.width, height2 = sh2.height;
    int scale = width / width2;
		//std::cout << "scale = " << scale << "\n";

    if ((!(scale == 1 || scale == 2 || scale == 4)) 
				|| (scale * width2 != width)
				|| (scale * height2 != height)){
			printf("   disp size = %4d x %4d\n", width2, height2);
			printf("GT disp size = %4d x %4d\n", width,  height);
			throw CError("GT disp size must be exactly 1, 2, or 4 * disp size");
    }

    int usemask = (msh.width > 0 && msh.height > 0);
    if (!usemask)
			throw CError("No mask image's been read yet!\n");
    if (usemask && (msh != sh))
			throw CError("mask image must have same size as GT!\n");

		// all region;
    int n_all = 0;
    int bad_all = 0;
    int invalid_all = 0;
    float err_all = 0;
		// non-occluded region;
    int invalid_noc = 0;
		int n_noc = 0;
    int bad_noc = 0;
    float err_noc = 0;
		// updated for mae and rmse error metric, on 2019/08/31
    float err2_all = 0; // square 
    float err2_noc = 0;
		

    for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				float gt = gtdisp.Pixel(x, y, 0);
				if (gt == INFINITY) // unknown
					continue;
				float d = scale * disp.Pixel(x / scale, y / scale, 0);
	      int valid = (d != INFINITY);
				if (valid) {
					float maxd = scale * maxdisp; // max disp range
		      d = __max(0, __min(maxd, d)); // clip disps to max disp range
				}
	      if (valid && rounddisp){
					d = round(d);
				} 
				float err = fabs(d - gt);
        
	      if (mask.Pixel(x,y,0) == 0){ // no ground truth;
					continue;
				}
				
				if (mask.Pixel(x, y, 0) == 255){ // non-occluded region;
					n_noc++;
					if (valid) {
						err_noc += err;
						err2_noc += err*err;

						if (err > badthresh) {
							bad_noc++;
						}
					} 
					else {// invalid (i.e. hole in sparse disp map)
						invalid_noc++;
					}
				}
				
				n_all++;
				if(valid){
					err_all += err;
					err2_all += err * err;

					if (err > badthresh){
						bad_all++;
					}
				}
				else {// invalid (i.e. hole in sparse disp map)
					invalid_all++;
				} 
			}/*end of width x*/
		}/*end of height y*/
    
		float badpercent_all =  (float)bad_all / (float)n_all;
    float avgErr_all = err_all / (float)(n_all - invalid_all); // CHANGED 10/14/2014 -- was: serr / n
		float badpercent_noc = (float) bad_noc / (float) n_noc;
		float avgErr_noc = err_noc /(float)(n_noc - invalid_noc);
		float rmse_all = sqrt(err2_all /(float)(n_all - invalid_all));
		float rmse_noc = sqrt(err2_noc /(float)(n_noc - invalid_noc));
#if 0
    float invalidpercent_all =  (float)invalid_all/(float)n_all;
    float totalbadpercent_all =  (float)(bad_all+invalid_all)/(float)n_all;
		printf("n_all = %d, n_noc = %d\n", n_all, n_noc);
    printf("%4.1f%(N-rate-all)  %6.2f%(bad-%2.1f-all)  %6.2f%(invalid-all)  %6.2f%(totalBad-all)  %6.2f(avgErr-all)\n",   
				100.0*n_all/(width * height), 100.0*badpercent_all, badthresh, 100.0*invalidpercent_all, 
				100.0*totalbadpercent_all, avgErr_all);
    float invalidpercent_noc =  (float)invalid_noc / (float)n_noc;
    float totalbadpercent_noc =  (float)(bad_noc+invalid_noc)/ (float)n_noc;
    printf("%4.1f%(N-rate-noc)  %6.2f%(bad-%2.1f-noc)  %6.2f%(invalid-noc)  %6.2f%(totalBad-noc)  %6.2f(avgErr-noc)\n",   
				100.0*n_noc/(width * height), 100.0*badpercent_noc, badthresh, 100.0*invalidpercent_noc, 
				100.0*totalbadpercent_noc, avgErr_noc);
#endif

		statistics[0] = badpercent_all;
		statistics[1] = avgErr_all;

		statistics[2] = badpercent_noc;
		statistics[3] = avgErr_noc;
		
		statistics[4] = rmse_all;
		statistics[5] = rmse_noc;
}

// C++ code
typedef std::vector<std::string> StringList;
class IMG_NAMES{
	private:
		StringList myvec;
	public:
		// constructor
		IMG_NAMES(){
			this -> myvec = StringList(0);
		}
	void set_img_names(boost::python::list & ns){
		for(int i = 0; i < len(ns); ++i){
			this -> myvec.push_back(boost::python::extract<std::string>(ns[i]));
		}
	}

	void show_img_names(){
		std:: cout << "Images :\n";
		for(auto i: this-> myvec){
			std::cout << i << ", ";
		}
		std:: cout << "\n";
	}

	StringList get_img_names(){
		return this -> myvec;
	}
};


PyObject * eval_disp_mbv3_1_img(
		const string & dispname,                // your predicted disparity results dir;
	  const string & disp_gt_trainingF_dir, 	// ground truth disparity dir to trainingF;
		float badthresh,                        // e.g., == 1.0 for Half resolution;
		const int &  ndisp,                      // max_disp in groud truth disparity;
		const int & rounddisp,
    const std::string & img_name           // 1 image to evaluate;
		){

		npy_intp * dims = new npy_intp[1];
		dims[0] = 4;
		PyObject * errs_A = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
	  //std::cout << "new PyObject errs_A, img_N = " << img_N << std::endl;
		float * err_result = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(errs_A)));
		
		//std::string dispname = disp_dir + img_name + "/" + disp_name; //"disp0.pfm"
		std::string gtdispname = disp_gt_trainingF_dir + img_name + "/disp0GT.pfm";
		// updated for middlebury additional data, due to the lack of mask0nocc.png file;
		std::string maskname = disp_gt_trainingF_dir + img_name + "/mask0nocc.png";
		bool readMask = true;
		if (!exists_file_test(maskname)){
			std::cout << "No mask0noc.png exists!\n";
			readMask = false;
		}
		
		//std::cout << "maxdisp = " << ndisp << ", "<< dispname << ", " << gtdispname << ", " << maskname << "\n";

		CFloatImage disp, gtdisp, gtdisp1;
		ReadImageVerb(disp, dispname.c_str(), verbose);
		ReadImageVerb(gtdisp, gtdispname.c_str(), verbose);
		CByteImage mask;
		if (readMask){
			//std::cout << "read mask0noc.png!\n";
			ReadImageVerb(mask, maskname.c_str(), verbose);
		}

		float statistics[4]= {-1.0, -1.0, -1.0, -1.0 };
		evaldisp(disp, gtdisp, mask, badthresh, ndisp, rounddisp, statistics);
		err_result[0] =  statistics[0];
		err_result[1] =  statistics[1];
		err_result[2] =  statistics[2];
		err_result[3] =  statistics[3];
		
		// release the memory
		disp.DeAllocate();
		gtdisp.DeAllocate();
		gtdisp1.DeAllocate();
		if (maskname.c_str())
			mask.DeAllocate();
		//exit(0);
		return errs_A;
}

PyObject * eval_disp_mbv3(
		const string & disp_dir, // your prediction disparity results dir;
	  const string & disp_gt_trainingF_dir, 	// ground truth disparity dir to trainingF;
		float badthresh, // e.g., == 1.0 for Half resolution;
		const string & disp_name, // e.g., == "_disp0" or "_disp0_post", or "_rf_disp0PKLS", and so on;
		PyObject * ndisps, // max_disp for each groud truth disparity 
		PyObject * rounddisps, // disparity is int or float, specified in calib.txt files;
    boost::python::list img_names // img_names to evaluate;
		){

		/*vector<std::string> mb_v3_imgs_1 = {
			"Adirondack",
			"ArtL", 
			"Jadeplant", 
			"Motorcycle",
			"MotorcycleE", 
			"Piano", 
			"PianoL", 
			"Pipes", 
			"Playroom", 
			"Playtable", 
		 	"PlaytableP", 
			"Recycle", 
			"Shelves",
			"Teddy", 
			"Vintage", 
		};
		*/
    IMG_NAMES mb_v3; 
		mb_v3.set_img_names(img_names);
		//mb_v3.show_img_names();
		vector<std::string> mb_v3_imgs = mb_v3.get_img_names();

    PyArrayObject* ndispsA = reinterpret_cast<PyArrayObject*>(ndisps);
    PyArrayObject* rounddispsA = reinterpret_cast<PyArrayObject*>(rounddisps);
		//int32_t * p_ndisps = reinterpret_cast<int32_t*>(PyArray_DATA(ndispsA));
		int * p_ndisps = reinterpret_cast<int*>(PyArray_DATA(ndispsA));
		int * p_rounddisps = reinterpret_cast<int*>(PyArray_DATA(rounddispsA));

		const int img_N = mb_v3_imgs.size();
		npy_intp * dims = new npy_intp[1];
		dims[0] = (4+2)*img_N;
		PyObject * errs_A = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
	  //std::cout << "new PyObject errs_A, img_N = " << img_N << std::endl;
		float * err_result = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(errs_A)));
		for (int i = 0; i < img_N; i++){
			std::string dispname = disp_dir + mb_v3_imgs[i] + "/" + disp_name + ".pfm";
			// check if a file exists or not;
			// > see: https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c;
		  if (!exists_file_test(dispname)){
				//std::cout << "Not exist : " << dispname << ", changed it to : ";
				
				//dispname = disp_dir + mb_v3_imgs[i] + "_" + disp_name;
				//Updated: delete "_" here;
				dispname = disp_dir + mb_v3_imgs[i] + disp_name + ".pfm";
				//std::cout << dispname << " ";
				if (!exists_file_test(dispname)){
					std::cout << "loading " << dispname << " ... but failed!\n"; 
				}
			}
      
			std::string gtdispname = disp_gt_trainingF_dir + mb_v3_imgs[i] + "/disp0GT.pfm";
			std::string maskname = disp_gt_trainingF_dir + mb_v3_imgs[i] + "/mask0nocc.png";
			bool readMask = true;
			// updated for middlebury additional data, due to the lack of mask0nocc.png file;
			if (!exists_file_test(maskname)){
			  std::cout << "No mask0noc.png exists!\n";
				readMask = false;
			}
	    int maxdisp = p_ndisps[i];
	    int rounddisp = p_rounddisps[i];
      //std::cout << "maxdisp = " << maxdisp << ", rounddisp = " << rounddisp << ", "<< dispname << ", " << gtdispname << ", " << maskname << "\n";

	    CFloatImage disp, gtdisp, gtdisp1;
	    ReadImageVerb(disp, dispname.c_str(), verbose);
	    ReadImageVerb(gtdisp, gtdispname.c_str(), verbose);
	    CByteImage mask;
	    if (readMask){
				//std::cout << "read mask0noc.png!\n";
				ReadImageVerb(mask, maskname.c_str(), verbose);
			}
	    
      //float statistics[4]= {-1.0, -1.0, -1.0, -1.0 };
			// updated for rmse metric on 2019/08/31;
	    float statistics[4 + 2]= {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0 };
			evaldisp(disp, gtdisp, mask, badthresh, maxdisp, rounddisp, statistics);
#if 0
			printf("Processing: %10s, %6.2f%(bad-%2.1f-all), %6.2f(avgErr-all), %6.2f%(bad-%2.1f-noc), %6.2f(avgErr-nov)\n", 
					mb_v3_imgs[i].c_str(), statistics[0]*100.0, badthresh, 
					statistics[1], statistics[2]*100.0, badthresh, statistics[3]);
#endif
			err_result[6*i]     =  statistics[0];
			err_result[6*i + 1] =  statistics[1];
			err_result[6*i + 2] =  statistics[2];
			err_result[6*i + 3] =  statistics[3];
			err_result[6*i + 4] =  statistics[4];
			err_result[6*i + 5] =  statistics[5];
      
			// release the memory
			disp.DeAllocate();
			gtdisp.DeAllocate();
			gtdisp1.DeAllocate();
	    if (maskname.c_str())
				mask.DeAllocate();

			//exit(0);
		
		}/*end of each image*/
		return errs_A;
}

BOOST_PYTHON_MODULE(libevaldisp_mbv3){
	/* for Boost <= version 1.63*/
	//numeric::array::set_module_and_type("numpy", "ndarray");
  /* for Boost > version 1.63*/
	//np::initialize();
#if BOOST_VERSION >= 106300 // >= 1.63.0
	np::initialize();
#else
	np::array::set_module_and_type("numpy", "ndarray");
#endif
	def("evaluate_mbv3", eval_disp_mbv3);
	def("evaluate_mbv3_1_img", eval_disp_mbv3_1_img);
	/* Error: return-statement with a value, in function returning 'void' [-fpermissive]
	 *        #define NUMPY_IMPORT_ARRAY_RETVAL NULL
	 * > See solution: https://github.com/numpy/numpy/issues/10486
	 * > 1) Solution1: Okay, so the issue occurs only on py2 + py3c (the initialization function is nonstandard, and has py3 semantics). Solution appears to be to use `import_array1()`;
	 * > 2) Solution2: Or call _import_array(), which allows you more control than just `return`;
	 */ 
  //import_array(); // work well for Python2.7;
	import_array1(); // work well for python3.7;
	//_import_array(); // work well for python3.7;
}

#include<Python.h>
#include<structmember.h>
#include<math.h>
#include<complex.h>
#include<string.h>
#include<time.h>
#include<numpy/arrayobject.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION



static PyObject* precompute_W2(PyObject* self, PyObject* args);
static PyObject* precompute_W3(PyObject* self, PyObject* args);
static PyObject* precompute_Evi(PyObject* self, PyObject* args);
static PyObject* precompute_d2chi2(PyObject* self, PyObject* args);

static int not_doublevector(PyArrayObject *vec);
static int not_complexvector(PyArrayObject *vec);
static int not_double_arr_nd(PyArrayObject *vec,int ndim);
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
double **ptrvector(long n);
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin);



/* ==== Create 1D Carray from PyArray ======================
 Assumes PyArray is contiguous in memory.             */
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
int i,n;

n=arrayin->dimensions[0];
return (double *) arrayin->data;  /* pointer to arrayin data as double */
}


// the following 2 functions are from/inspired by: http://wiki.scipy.org/Cookbook/C_Extensions/NumPy_arrays
// purpose of the functions: check if the vectors have the correct data type.
static int not_doublevector(PyArrayObject *vec)  {
  if (vec->descr->type_num != NPY_DOUBLE || vec->nd != 1)  {
    PyErr_SetString(PyExc_ValueError,
      "In not_doublevector: array must be of type Float and 1 dimensional (n).");
    return 1;  }
  return 0;
}

static int  not_complexvector(PyArrayObject *vec)  {
  if (vec->descr->type_num != NPY_CDOUBLE || vec->nd != 1)  {
    PyErr_SetString(PyExc_ValueError,
      "In not_complexvector: array must be of type Integer and 1 dimensional (n).");
    return 1;  }
  return 0;
}


static int  not_double_arr_nd(PyArrayObject *vec,int ndim)  {
  if (vec->descr->type_num != NPY_DOUBLE || vec->nd != ndim)  {
    PyErr_SetString(PyExc_ValueError,
      "In not_complex_arr2d: array must be of type Integer and n dimensional (n).");
    return 1;  }
  return 0;
}

/* ==== Create Carray from PyArray ======================
 * Assumes PyArray is contiguous in memory.
 * Memory is allocated!                                    */
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin)  {
double **c, *a;
int i,n,m;

n=arrayin->dimensions[0];
m=arrayin->dimensions[1];
c=ptrvector(n);
a=(double *) arrayin->data;  /* pointer to arrayin data as double */
for ( i=0; i<n; i++)  {
                c[i]=a+i*m;  }
return c;
}

/* ==== Allocate a double *vector (vec of pointers) ======================
 Memory is Allocated!  See void free_Carray(double ** )                  */
double **ptrvector(long n)  {
double **v;
v=(double **)malloc((size_t) (n*sizeof(double)));
if (!v)   {
        printf("In **ptrvector. Allocation of memory for double array failed.");
        exit(0);  }
return v;
}

/*------------------------------------------------------------------------------------------------------*/
/*------------------------------Functions to be accessed from Python------------------------------------*/
/*------------------------------------------------------------------------------------------------------*/






static PyObject* precompute_W2(PyObject* self, PyObject* args)
{
  PyArrayObject* W2_py;
  PyArrayObject* E_py;
  PyArrayObject* U_py;
  PyArrayObject* Xi_py;
  PyArrayObject* V_py;
  PyArrayObject* dw_py;
  PyArrayObject* model_py;
  double **W2, **U, **V;
  double *E, *Xi, *dw, *model;
  int m,l,k,n;
  int nsv,nw,niw;

  if (!PyArg_ParseTuple(args,"O!O!O!O!O!O!O!",&PyArray_Type,&W2_py,
                                        &PyArray_Type,&E_py,
                                        &PyArray_Type,&U_py,
                                        &PyArray_Type,&Xi_py,
                                        &PyArray_Type,&V_py,
                                        &PyArray_Type,&dw_py,
                                        &PyArray_Type,&model_py)) {
    printf("Arguments of precompute_W2 could not be parsed.\n");
    return NULL;
  }

  if (W2_py==NULL) return NULL;
  if (E_py==NULL) return NULL;
  if (U_py==NULL) return NULL;
  if (Xi_py==NULL) return NULL;
  if (V_py==NULL) return NULL;
  if (dw_py==NULL) return NULL;
  if (model_py==NULL) return NULL;
  
  if (not_double_arr_nd(W2_py,2)) return NULL;
  if (not_double_arr_nd(U_py,2)) return NULL;
  if (not_double_arr_nd(V_py,2)) return NULL;
  if (not_double_arr_nd(E_py,1)) return NULL;
  if (not_double_arr_nd(Xi_py,1)) return NULL;
  if (not_double_arr_nd(dw_py,1)) return NULL;
  if (not_double_arr_nd(model_py,1)) return NULL;

  npy_intp* shape_W2  =  PyArray_DIMS(W2_py);
  npy_intp* shape_E = PyArray_DIMS(E_py);
  nsv=shape_W2[0];
  nw=shape_W2[1];
  niw=shape_E[0];


  W2=pymatrix_to_Carrayptrs(W2_py);
  U=pymatrix_to_Carrayptrs(U_py);
  V=pymatrix_to_Carrayptrs(V_py);
  E=pyvector_to_Carrayptrs(E_py);
  Xi=pyvector_to_Carrayptrs(Xi_py);
  dw=pyvector_to_Carrayptrs(dw_py);
  model=pyvector_to_Carrayptrs(model_py);


  for(m=0;m<nsv;m++) {
    for(l=0;l<nw;l++) {
      for(k=0;k<niw;k++) {
        for(n=0;n<nsv;n++) {
          W2[m][l]+=E[k]*U[k][m]*Xi[m]*U[k][n]*Xi[n]*V[l][n]*dw[l]*model[l];
        }
      }
    }
  }
  
  

  Py_RETURN_NONE;

}


static PyObject* precompute_d2chi2(PyObject* self, PyObject* args)
{

  PyArrayObject* d2chi2_py;
  PyArrayObject* kernel_py;
  PyArrayObject* E_py;
  PyArrayObject* dw_py;
  double **d2chi2, **kernel;
  double *E, *dw;
  int i,j,k;
  int nw,niw;

  if (!PyArg_ParseTuple(args,"O!O!O!O!",&PyArray_Type,&d2chi2_py,
                                              &PyArray_Type,&kernel_py,
                                              &PyArray_Type,&dw_py,
                                              &PyArray_Type,&E_py)) {
    printf("Arguments of precompute_d2chi2 could not be parsed.\n");
    return NULL;
  }

  if (d2chi2_py==NULL) return NULL;
  if (E_py==NULL) return NULL;
  if (kernel_py==NULL) return NULL;
  if (dw_py==NULL) return NULL;
  
  if (not_double_arr_nd(d2chi2_py,2)) return NULL;
  if (not_double_arr_nd(kernel_py,2)) return NULL;
  if (not_double_arr_nd(E_py,1)) return NULL;
  if (not_double_arr_nd(dw_py,1)) return NULL;

  npy_intp* shape_d2chi2  =  PyArray_DIMS(d2chi2_py);
  npy_intp* shape_E = PyArray_DIMS(E_py);
  nw=shape_d2chi2[0];
  niw=shape_E[0];


  d2chi2=pymatrix_to_Carrayptrs(d2chi2_py);
  kernel=pymatrix_to_Carrayptrs(kernel_py);
  E=pyvector_to_Carrayptrs(E_py);
  dw=pyvector_to_Carrayptrs(dw_py);


  for(i=0;i<nw;i++) {
    for(j=0;j<nw;j++) {
      d2chi2[i][j]=0.;
      for(k=0;k<niw;k++) {
          d2chi2[i][j]+=dw[i]*dw[j]*kernel[k][i]*kernel[k][j]*E[k];
      }
    }
  }
  



  Py_RETURN_NONE;

}

static PyObject* precompute_W3(PyObject* self, PyObject* args)
{

  Py_RETURN_NONE;

}

static PyObject* precompute_Evi(PyObject* self, PyObject* args)
{

  Py_RETURN_NONE;

}
/*------------------------------------------------------------------------------
        Start of Python-specific stuff.
        Various definitions to set up the interface between Python an C.*/

/*      Gerneral methods of the module. */
static PyMethodDef module_methods[] = {
    {"precompute_W2", (PyCFunction)precompute_W2, METH_VARARGS, ""},
    {"precompute_W3", (PyCFunction)precompute_W3, METH_VARARGS, ""},
    {"precompute_Evi", (PyCFunction)precompute_Evi, METH_VARARGS, ""},
    {"precompute_d2chi2", (PyCFunction)precompute_d2chi2, METH_VARARGS, ""},
    {NULL}  /* Sentinel */
};


/*      Initialization of the module.   */
PyMODINIT_FUNC initprecomp(void)
{
    PyObject* obj;

    obj = Py_InitModule3("precomp", module_methods, "Precomputation of coefficient matrices for my Maxent.");

        /*      Very important call, otherwise something crashes.       */
        import_array();

    if (obj == NULL)
      return;
}
/*      End of Python-specific stuff.
------------------------------------------------------------------------------*/

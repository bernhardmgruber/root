// Bindings
#include "CPyCppyy.h"
#include "TupleOfInstances.h"
#include "ProxyWrappers.h"


namespace {

typedef struct {
    PyObject_HEAD
    Cppyy::TCppType_t        ia_klass;
    void*                    ia_array_start;
    Py_ssize_t               ia_pos;
    Py_ssize_t               ia_len;
    Py_ssize_t               ia_stride;
} ia_iterobject;

static PyObject* ia_iternext(ia_iterobject* ia) {
    if (ia->ia_len != -1 && ia->ia_pos >= ia->ia_len) {
        ia->ia_pos = 0;      // debatable, but since the iterator is cached, this
        return nullptr;      //   allows for multiple conversions to e.g. a tuple
    } else if (ia->ia_stride == 0 && ia->ia_pos != 0) {
        PyErr_SetString(PyExc_ReferenceError, "no stride available for indexing");
        return nullptr;
    }
    PyObject* result = CPyCppyy::BindCppObjectNoCast(
        (char*)ia->ia_array_start + ia->ia_pos*ia->ia_stride, ia->ia_klass);
    ia->ia_pos += 1;
    return result;
}

static int ia_traverse(ia_iterobject*, visitproc, void*) {
    return 0;
}

static PyObject* ia_getsize(ia_iterobject* ia, void*) {
    return PyInt_FromSsize_t(ia->ia_len);
}

static int ia_setsize(ia_iterobject* ia, PyObject* pysize, void*) {
    Py_ssize_t size = PyInt_AsSsize_t(pysize);
    if (size == (Py_ssize_t)-1 && PyErr_Occurred())
        return -1;
    ia->ia_len = size;
    return 0;
}

static PyGetSetDef ia_getset[] = {
    {(char*)"size", (getter)ia_getsize, (setter)ia_setsize,
      (char*)"set size of array to which this iterator refers", nullptr},
    {(char*)nullptr, nullptr, nullptr, nullptr, nullptr}
};

} // unnamed namespace


namespace CPyCppyy {

PyTypeObject InstanceArrayIter_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    (char*)"cppyy.instancearrayiter",  // tp_name
    sizeof(ia_iterobject),        // tp_basicsize
    0,
    (destructor)PyObject_GC_Del,  // tp_dealloc
    0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_HAVE_GC,       // tp_flags
    nullptr,
    (traverseproc)ia_traverse,    // tp_traverse
    nullptr, nullptr, 0,
    PyObject_SelfIter,            // tp_iter
    (iternextfunc)ia_iternext,    // tp_iternext
    nullptr, nullptr, ia_getset, nullptr, nullptr, nullptr, nullptr,
    0,                   // tp_getset
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
#if PY_VERSION_HEX >= 0x02030000
    , nullptr                           // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
    , 0                           // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
    , nullptr                           // tp_finalize
#endif
};


//= support for C-style arrays of objects ====================================
PyObject* TupleOfInstances_New(
    Cppyy::TCppObject_t address, Cppyy::TCppType_t klass, dim_t ndims, dims_t dims)
{
// recursively set up tuples of instances on all dimensions
    if (ndims == -1 /* unknown shape */ || dims[0] == -1 /* unknown size */) {
    // no known length ... return an iterable object and let the user figure it out
        ia_iterobject* ia = PyObject_GC_New(ia_iterobject, &InstanceArrayIter_Type);
        if (!ia) return nullptr;

        ia->ia_klass       = klass;
        ia->ia_array_start = address;
        ia->ia_pos         = 0;
        ia->ia_len         = -1;
        ia->ia_stride      = Cppyy::SizeOf(klass);

        PyObject_GC_Track(ia);
        return (PyObject*)ia;
    } else if (1 < ndims) {
    // not the innermost dimension, descend one level
        int nelems = (int)dims[0];
        size_t block_size = 0;
        for (int i = 1; i < (int)ndims; ++i) block_size += (size_t)dims[i];
        block_size *= Cppyy::SizeOf(klass);

        PyObject* tup = PyTuple_New(nelems);
        for (int i = 0; i < nelems; ++i) {
            PyTuple_SetItem(tup, i, TupleOfInstances_New(
                (char*)address + i*block_size, klass, ndims-1, dims+1));
        }
        return tup;
    } else {
    // innermost dimension: construct tuple
        int nelems = (int)dims[0];
        size_t block_size = Cppyy::SizeOf(klass);
        if (block_size == 0) {
            PyErr_Format(PyExc_TypeError,
                "can not determine size of type \"%s\" for array indexing",
                Cppyy::GetScopedFinalName(klass).c_str());
            return nullptr;
        }

    // TODO: the extra copy is inefficient, but it appears that the only way to
    // initialize a subclass of a tuple is through a sequence
        PyObject* tup = PyTuple_New(nelems);
        for (int i = 0; i < nelems; ++i) {
        // TODO: there's an assumption here that there is no padding, which is bound
        // to be incorrect in certain cases
            PyTuple_SetItem(tup, i,
                BindCppObjectNoCast((char*)address + i*block_size, klass));
        // Note: objects are bound as pointers, yet since the pointer value stays in
        // place, updates propagate just as if they were bound by-reference
        }

        PyObject* args = PyTuple_New(1);
        Py_INCREF(tup); PyTuple_SET_ITEM(args, 0, tup);
        PyObject* arr = PyTuple_Type.tp_new(&TupleOfInstances_Type, args, nullptr);
        if (PyErr_Occurred()) PyErr_Print();

        Py_DECREF(args);
        // tup ref eaten by SET_ITEM on args

        return arr;
    }

// never get here
    return nullptr;
}

//= CPyCppyy custom tuple-like array type ====================================
PyTypeObject TupleOfInstances_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    (char*)"cppyy.InstancesArray", // tp_name
    0,                             // tp_basicsize
    0,                             // tp_itemsize
    nullptr,                             // tp_dealloc
    0,                             // tp_print
    nullptr,                             // tp_getattr
    nullptr,                             // tp_setattr
    nullptr,                             // tp_compare
    nullptr,                             // tp_repr
    nullptr,                             // tp_as_number
    nullptr,                             // tp_as_sequence
    nullptr,                             // tp_as_mapping
    nullptr,                             // tp_hash
    nullptr,                             // tp_call
    nullptr,                             // tp_str
    nullptr,                             // tp_getattro
    nullptr,                             // tp_setattro
    nullptr,                             // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES |
        Py_TPFLAGS_BASETYPE,       // tp_flags
    (char*)"array of C++ instances",    // tp_doc
    nullptr,                             // tp_traverse
    nullptr,                             // tp_clear
    nullptr,                             // tp_richcompare
    0,                             // tp_weaklistoffset
    nullptr,                             // tp_iter
    nullptr,                             // tp_iternext
    nullptr,                             // tp_methods
    nullptr,                             // tp_members
    nullptr,                             // tp_getset
    &PyTuple_Type,                 // tp_base
    nullptr,                             // tp_dict
    nullptr,                             // tp_descr_get
    nullptr,                             // tp_descr_set
    0,                             // tp_dictoffset
    nullptr,                             // tp_init
    nullptr,                             // tp_alloc
    nullptr,                             // tp_new
    nullptr,                             // tp_free
    nullptr,                             // tp_is_gc
    nullptr,                             // tp_bases
    nullptr,                             // tp_mro
    nullptr,                             // tp_cache
    nullptr,                             // tp_subclasses
    nullptr                              // tp_weaklist
#if PY_VERSION_HEX >= 0x02030000
    , nullptr                            // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
    , 0                            // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
    , nullptr                            // tp_finalize
#endif
};

} // namespace CPyCppyy

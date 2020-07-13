/*****************************************************************************
 * Project: RooFit                                                           *
 *                                                                           *
 * This code was autogenerated by RooClassFactory                            *
 *****************************************************************************/

/** \class RooParamHistFunc
 *  \ingroup Roofit
 * A histogram function that assigns scale parameters to every bin. Instead of the bare bin contents,
 * it therefore yields:
 * \f[
 *  \gamma_{i} * \mathrm{bin}_i
 * \f]
 *
 * The \f$ \gamma_i \f$ can therefore be used to parametrise statistical uncertainties of the histogram
 * template. In conjuction with a constraint term, this can be used to implement the Barlow-Beeston method.
 * The constraint can be implemented using RooHistConstraint.
 *
 * See also the tutorial rf709_BarlowBeeston.C
 */

#include "Riostream.h"
#include "RooParamHistFunc.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "RooRealVar.h"
#include <math.h>
#include "TMath.h"

using namespace std ;

ClassImp(RooParamHistFunc);

////////////////////////////////////////////////////////////////////////////////
/// Populate x with observables

RooParamHistFunc::RooParamHistFunc(const char *name, const char *title, RooDataHist& dh, Bool_t paramRelative) :
  RooAbsReal(name,title),
  _x("x","x",this),
  _p("p","p",this),
  _dh(dh),
  _relParam(paramRelative)
{
  _x.add(*_dh.get()) ;

  // Now populate p with parameters
  RooArgSet allVars ;
  for (Int_t i=0 ; i<_dh.numEntries() ; i++) {
    _dh.get(i) ;

    const char* vname = Form("%s_gamma_bin_%i",GetName(),i) ;
    RooRealVar* var = new RooRealVar(vname,vname,0,1000) ;
    var->setVal(_relParam ? 1 : _dh.weight()) ;
    var->setError(_relParam ? 1 / sqrt(_dh.weight()) : sqrt(_dh.weight()));
    var->setConstant(kTRUE) ;
    allVars.add(*var) ;
    _p.add(*var) ;
  }
  addOwnedComponents(allVars) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Populate x with observables

RooParamHistFunc::RooParamHistFunc(const char *name, const char *title, const RooAbsArg& /*x*/, RooDataHist& dh, Bool_t paramRelative) :
  RooAbsReal(name,title),
  _x("x","x",this),
  _p("p","p",this),
  _dh(dh),
  _relParam(paramRelative)
{
  _x.add(*_dh.get()) ;

  // Now populate p with parameters
  RooArgSet allVars ;
  for (Int_t i=0 ; i<_dh.numEntries() ; i++) {
    _dh.get(i) ;
    const char* vname = Form("%s_gamma_bin_%i",GetName(),i) ;
    RooRealVar* var = new RooRealVar(vname,vname,0,1000) ;
    var->setVal(_relParam ? 1 : _dh.weight()) ;
    var->setError(_relParam ? 1 / sqrt(_dh.weight()) : sqrt(_dh.weight()));
    var->setConstant(kTRUE) ;
    allVars.add(*var) ;
    _p.add(*var) ;
  }
  addOwnedComponents(allVars) ;
}

////////////////////////////////////////////////////////////////////////////////

RooParamHistFunc::RooParamHistFunc(const char *name, const char *title, RooDataHist& dh, const RooParamHistFunc& paramSource, Bool_t paramRelative) :
  RooAbsReal(name,title),
  _x("x","x",this),
  _p("p","p",this),
  _dh(dh),
  _relParam(paramRelative)
{
  // Populate x with observables
  _x.add(*_dh.get()) ;

  // Now populate p with existing parameters
  _p.add(paramSource._p) ;
}

////////////////////////////////////////////////////////////////////////////////

RooParamHistFunc::RooParamHistFunc(const RooParamHistFunc& other, const char* name) :
  RooAbsReal(other,name),
  _x("x",this,other._x),
  _p("p",this,other._p),
  _dh(other._dh),
  _relParam(other._relParam)
{
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooParamHistFunc::evaluate() const
{
  Int_t idx = ((RooDataHist&)_dh).getIndex(_x,kTRUE) ;
  Double_t ret = ((RooAbsReal*)_p.at(idx))->getVal() ;
  if (_relParam) {
    Double_t nom = getNominal(idx) ;
    ret *= nom ;
  }
  return  ret ;
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooParamHistFunc::getActual(Int_t ibin)
{
  return ((RooAbsReal&)_p[ibin]).getVal() ;
}

////////////////////////////////////////////////////////////////////////////////

void RooParamHistFunc::setActual(Int_t ibin, Double_t newVal)
{
  ((RooRealVar&)_p[ibin]).setVal(newVal) ;
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooParamHistFunc::getNominal(Int_t ibin) const
{
  _dh.get(ibin) ;
  return _dh.weight() ;
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooParamHistFunc::getNominalError(Int_t ibin) const
{
  _dh.get(ibin) ;
  return _dh.weightError() ;
}

////////////////////////////////////////////////////////////////////////////////
/// Return sampling hint for making curves of (projections) of this function
/// as the recursive division strategy of RooCurve cannot deal efficiently
/// with the vertical lines that occur in a non-interpolated histogram

list<Double_t>* RooParamHistFunc::plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const
{
  // Check that observable is in dataset, if not no hint is generated
  RooAbsLValue* lvarg = dynamic_cast<RooAbsLValue*>(_dh.get()->find(obs.GetName())) ;
  if (!lvarg) {
    return nullptr ;
  }

  // Retrieve position of all bin boundaries
  const RooAbsBinning* binning = lvarg->getBinningPtr(nullptr) ;
  Double_t* boundaries = binning->array() ;

  list<Double_t>* hint = new list<Double_t> ;

  // Widen range slightly
  xlo = xlo - 0.01*(xhi-xlo) ;
  xhi = xhi + 0.01*(xhi-xlo) ;

  Double_t delta = (xhi-xlo)*1e-8 ;

  // Construct array with pairs of points positioned epsilon to the left and
  // right of the bin boundaries
  for (Int_t i=0 ; i<binning->numBoundaries() ; i++) {
    if (boundaries[i]>=xlo && boundaries[i]<=xhi) {
      hint->push_back(boundaries[i]-delta) ;
      hint->push_back(boundaries[i]+delta) ;
    }
  }

  return hint ;
}

////////////////////////////////////////////////////////////////////////////////
/// Return sampling hint for making curves of (projections) of this function
/// as the recursive division strategy of RooCurve cannot deal efficiently
/// with the vertical lines that occur in a non-interpolated histogram

std::list<Double_t>* RooParamHistFunc::binBoundaries(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const
{
  // Check that observable is in dataset, if not no hint is generated
  RooAbsLValue* lvarg = dynamic_cast<RooAbsLValue*>(_dh.get()->find(obs.GetName())) ;
  if (!lvarg) {
    return nullptr ;
  }

  // Retrieve position of all bin boundaries
  const RooAbsBinning* binning = lvarg->getBinningPtr(nullptr) ;
  Double_t* boundaries = binning->array() ;

  list<Double_t>* hint = new list<Double_t> ;

  // Construct array with pairs of points positioned epsilon to the left and
  // right of the bin boundaries
  for (Int_t i=0 ; i<binning->numBoundaries() ; i++) {
    if (boundaries[i]>=xlo && boundaries[i]<=xhi) {
      hint->push_back(boundaries[i]) ;
    }
  }

  return hint ;
}

////////////////////////////////////////////////////////////////////////////////
/// Advertise that all integrals can be handled internally.

Int_t RooParamHistFunc::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars,
                  const RooArgSet* /*normSet*/, const char* /*rangeName*/) const
{
  // Simplest scenario, integrate over all dependents
  RooAbsCollection *allVarsCommon = allVars.selectCommon(_x) ;
  Bool_t intAllObs = (allVarsCommon->getSize()==_x.getSize()) ;
  delete allVarsCommon ;
  if (intAllObs && matchArgs(allVars,analVars,_x)) {
    return 1 ;
  }

  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////
/// Implement analytical integrations by doing appropriate weighting from  component integrals
/// functions to integrators of components

Double_t RooParamHistFunc::analyticalIntegralWN(Int_t code, const RooArgSet* /*normSet2*/,const char* /*rangeName*/) const
{
  R__ASSERT(code==1) ;

  Double_t ret(0) ;
  Int_t i(0) ;
  for (const auto param : _p) {
    auto p = static_cast<const RooAbsReal*>(param);
    Double_t bin = p->getVal() ;
    if (_relParam) bin *= getNominal(i++) ;
    ret += bin ;
  }

  // WVE fix this!!! Assume uniform binning for now!
  Double_t binV(1) ;
  for (const auto obs : _x) {
    auto xx = static_cast<const RooRealVar*>(obs);
    binV *= (xx->getMax()-xx->getMin())/xx->numBins() ;
  }

  return ret*binV ;
}

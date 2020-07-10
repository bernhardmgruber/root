// @(#)root/geom:$Id$
// Author: Andrei Gheata   24/10/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoTrd1
#define ROOT_TGeoTrd1

#include "TGeoBBox.h"

class TGeoTrd1 : public TGeoBBox
{
protected:
   // data members
   Double_t              fDx1; // half length in X at lower Z surface (-dz)
   Double_t              fDx2; // half length in X at higher Z surface (+dz)
   Double_t              fDy;  // half length in Y
   Double_t              fDz;  // half length in Z

   // methods

public:
   // constructors
   TGeoTrd1();
   TGeoTrd1(Double_t dx1, Double_t dx2, Double_t dy, Double_t dz);
   TGeoTrd1(const char *name, Double_t dx1, Double_t dx2, Double_t dy, Double_t dz);
   TGeoTrd1(Double_t *params);
   // destructor
   ~TGeoTrd1() override;
   // methods

   Double_t Capacity() const override;
   void     ComputeBBox() override;
   void     ComputeNormal(const Double_t *point, const Double_t *dir, Double_t *norm) override;
   void     ComputeNormal_v(const Double_t *points, const Double_t *dirs, Double_t *norms, Int_t vecsize) override;
   Bool_t   Contains(const Double_t *point) const override;
   void     Contains_v(const Double_t *points, Bool_t *inside, Int_t vecsize) const override;
   Double_t DistFromInside(const Double_t *point, const Double_t *dir, Int_t iact = 1, Double_t step = TGeoShape::Big(),
                           Double_t *safe = 0) const override;
   void     DistFromInside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize,
                             Double_t *step) const override;
   Double_t DistFromOutside(const Double_t *point, const Double_t *dir, Int_t iact = 1,
                            Double_t step = TGeoShape::Big(), Double_t *safe = 0) const override;
   void     DistFromOutside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize,
                              Double_t *step) const override;
   TGeoVolume *
   Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, Double_t start, Double_t step) override;
   Double_t              GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const override;
   void                  GetBoundingCylinder(Double_t *param) const override;
   Int_t                 GetByteCount() const override { return 52; }
   Double_t              GetDx1() const {return fDx1;}
   Double_t              GetDx2() const {return fDx2;}
   Double_t              GetDy() const  {return fDy;}
   Double_t              GetDz() const  {return fDz;}
   Int_t
   GetFittingBox(const TGeoBBox *parambox, TGeoMatrix *mat, Double_t &dx, Double_t &dy, Double_t &dz) const override;
   TGeoShape *           GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const override;
   void                  GetVisibleCorner(const Double_t *point, Double_t *vertex, Double_t *normals) const;
   void                  GetOppositeCorner(const Double_t *point, Int_t inorm, Double_t *vertex, Double_t *normals) const;
   void                  InspectShape() const override;
   Bool_t                IsCylType() const override { return kFALSE; }
   Double_t              Safety(const Double_t *point, Bool_t in = kTRUE) const override;
   void Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void SetDimensions(Double_t *param) override;
   void SetPoints(Double_t *points) const override;
   void SetPoints(Float_t *points) const override;
   void                  SetVertex(Double_t *vertex) const;
   void                  Sizeof3D() const override;

   ClassDef(TGeoTrd1, 1)         // TRD1 shape class
};

#endif

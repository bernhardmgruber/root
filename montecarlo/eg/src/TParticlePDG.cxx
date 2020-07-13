// @(#)root/eg:$Id$
// Author: Pasha Murat   12/02/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class  TParticlePDG
    \ingroup eg

Description of the static properties of a particle.

The class is typically generated by the TDatabasePDG class.
It is referenced by the dynamic particle class TParticle.
\verbatim
 Int_t            fPdgCode;          // PDG code of the particle
 Double_t         fMass;             // particle mass in GeV
 Double_t         fCharge;           // charge in units of |e|/3
 Double_t         fLifetime;         // proper lifetime in seconds
 Double_t         fWidth;            // total width in GeV
 Int_t            fParity;           // parity
 Double_t         fSpin;             // spin
 Double_t         fIsospin;          // isospin
 Double_t         fI3;               // i3
 Int_t            fStrangeness;      // flavours are defined if i3 != -1
 Int_t            fCharm;            // 1 or -1 for C-particles,  0 for others
 Int_t            fBeauty;
 Int_t            fTop;
 Int_t            fY;                // X,Y: quantum numbers for the 4-th generation
 Int_t            fX;
 Int_t            fStable;           // 1 if stable, 0 otherwise

 TObjArray*       fDecayList;        // array of decay channels

 TString          fParticleClass;    // lepton, meson etc

 Int_t            fTrackingCode;     // G3 tracking code of the particle
 TParticlePDG*    fAntiParticle;     // pointer to antiparticle
\endverbatim
*/

#include "TDecayChannel.h"
#include "TParticlePDG.h"
#include "TDatabasePDG.h"

ClassImp(TParticlePDG);

////////////////////////////////////////////////////////////////////////////////
///default constructor

TParticlePDG::TParticlePDG()
{
   fPdgCode      = 0;
   fMass         = 0;
   fCharge       = 0;
   fLifetime     = 0;
   fWidth        = 0;
   fParity       = 0;
   fSpin         = 0;
   fIsospin      = 0;
   fI3           = 0;
   fStrangeness  = 0;
   fCharm        = 0;
   fBeauty       = 0;
   fTop          = 0;
   fY            = 0;
   fX            = 0;
   fStable       = 0;
   fDecayList    = nullptr;
   fTrackingCode = 0;
   fAntiParticle = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
///constructor

TParticlePDG::TParticlePDG(const char* Name, const char* Title, Double_t aMass,
                           Bool_t aStable, Double_t aWidth, Double_t aCharge,
                           const char* aParticleClass, Int_t aPdgCode, Int_t Anti,
                           Int_t aTrackingCode)
  : TNamed(Name,Title)
{
   // empty for the time  being
   fLifetime      = 0;
   fParity        = 0;
   fSpin          = 0;
   fIsospin       = 0;
   fI3            = 0;
   fStrangeness   = 0;
   fCharm         = 0;
   fBeauty        = 0;
   fTop           = 0;
   fY             = 0;
   fX             = 0;
   fStable        = 0;

   fMass          = aMass;
   fStable        = aStable;
   fWidth         = aWidth;
   fCharge        = aCharge;
   fParticleClass = aParticleClass;
   fPdgCode       = aPdgCode;
   fTrackingCode  = aTrackingCode;
   fDecayList     = nullptr;
   if (Anti) fAntiParticle = this;
   else      fAntiParticle = nullptr;

   const Double_t kHbar = 6.58211889e-25; // GeV s
   if (fWidth != 0.) fLifetime = kHbar / fWidth;
}

////////////////////////////////////////////////////////////////////////////////
///copy constructor

TParticlePDG::TParticlePDG(const TParticlePDG& pdg) :
  TNamed(pdg),
  fPdgCode(pdg.fPdgCode),
  fMass(pdg.fMass),
  fCharge(pdg.fCharge),
  fLifetime(pdg.fLifetime),
  fWidth(pdg.fWidth),
  fParity(pdg.fParity),
  fSpin(pdg.fSpin),
  fIsospin(pdg.fIsospin),
  fI3(pdg.fI3),
  fStrangeness(pdg.fStrangeness),
  fCharm(pdg.fCharm),
  fBeauty(pdg.fBeauty),
  fTop(pdg.fTop),
  fY(pdg.fY),
  fX(pdg.fX),
  fStable(pdg.fStable),
  fDecayList(pdg.fDecayList),
  fParticleClass(pdg.fParticleClass),
  fTrackingCode(pdg.fTrackingCode),
  fAntiParticle(pdg.fAntiParticle)
{
}

////////////////////////////////////////////////////////////////////////////////
///assignement operator

TParticlePDG& TParticlePDG::operator=(const TParticlePDG& pdg)
{
   if(this!=&pdg) {
      TNamed::operator=(pdg);
      fPdgCode=pdg.fPdgCode;
      fMass=pdg.fMass;
      fCharge=pdg.fCharge;
      fLifetime=pdg.fLifetime;
      fWidth=pdg.fWidth;
      fParity=pdg.fParity;
      fSpin=pdg.fSpin;
      fIsospin=pdg.fIsospin;
      fI3=pdg.fI3;
      fStrangeness=pdg.fStrangeness;
      fCharm=pdg.fCharm;
      fBeauty=pdg.fBeauty;
      fTop=pdg.fTop;
      fY=pdg.fY;
      fX=pdg.fX;
      fStable=pdg.fStable;
      fDecayList=pdg.fDecayList;
      fParticleClass=pdg.fParticleClass;
      fTrackingCode=pdg.fTrackingCode;
      fAntiParticle=pdg.fAntiParticle;
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
///destructor

TParticlePDG::~TParticlePDG() {
   if (fDecayList) {
      fDecayList->Delete();
      delete fDecayList;
   }
}


////////////////////////////////////////////////////////////////////////////////
/// add new decay channel, Particle owns those...

Int_t TParticlePDG::AddDecayChannel(Int_t        Type,
                                    Double_t     BranchingRatio,
                                    Int_t        NDaughters,
                                    Int_t*       DaughterPdgCode)
{
   Int_t n = NDecayChannels();
   if (NDecayChannels() == 0) {
      fDecayList = new TObjArray(5);
   }
   TDecayChannel* dc = new TDecayChannel(n,Type,BranchingRatio,NDaughters,
                                        DaughterPdgCode);
   fDecayList->Add(dc);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// return pointer to decay channel object at index i

TDecayChannel* TParticlePDG::DecayChannel(Int_t i)
{
   return (TDecayChannel*) fDecayList->At(i);
}

////////////////////////////////////////////////////////////////////////////////
/// print the list of decays

void TParticlePDG::PrintDecayChannel(TDecayChannel* dc, Option_t* option) const
{
   if (strstr(option,"banner")) {
                                // print banner

      printf(" Channel Code BranchingRatio Nd  ");
      printf(" ...................Daughters.................... \n");
   }
   if (strstr(option,"data")) {

      TDatabasePDG* db = TDatabasePDG::Instance();

      printf("%7i %5i %12.5e %5i  ",
           dc->Number(),
           dc->MatrixElementCode(),
           dc->BranchingRatio(),
           dc->NDaughters());

      for (int i=0; i<dc->NDaughters(); i++) {
         int ic = dc->DaughterPdgCode(i);
         TParticlePDG* p = db->GetParticle(ic);
         printf(" %15s(%8i)",p->GetName(),ic);
      }
      printf("\n");
   }
}


////////////////////////////////////////////////////////////////////////////////
///
///  Print the entire information of this kind of particle
///

void TParticlePDG::Print(Option_t *) const
{
   printf("%-20s  %6d\t",GetName(),fPdgCode);
   if (!fStable) {
      printf("Mass:%9.4f Width (GeV):%11.4e\tCharge: %5.1f\n",
              fMass, fWidth, fCharge);
   } else {
      printf("Mass:%9.4f Width (GeV): Stable\tCharge: %5.1f\n",
              fMass, fCharge);
   }
   if (fDecayList) {
      int banner_printed = 0;
      TIter next(fDecayList);
      TDecayChannel* dc;
      while ((dc = (TDecayChannel*)next())) {
         if (! banner_printed) {
            PrintDecayChannel(dc,"banner");
            banner_printed = 1;
         }
         PrintDecayChannel(dc,"data");
      }
   }
}


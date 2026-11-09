#include "hrr_derivatives.h"
namespace pyscfad_sgx {
HRRDerivativeKernel select_hrr_derivative(int a,int b,bool same) {
switch((a*5+b)*2+same) {
case 0: return HD_S_S;
case 1: return HD_S_S_SAMECENTER;
case 10: return HD_P_S;
case 11: return HD_P_S_SAMECENTER;
case 12: return HD_P_P;
case 13: return HD_P_P_SAMECENTER;
case 20: return HD_D_S;
case 21: return HD_D_S_SAMECENTER;
case 22: return HD_D_P;
case 23: return HD_D_P_SAMECENTER;
case 24: return HD_D_D;
case 25: return HD_D_D_SAMECENTER;
case 30: return HD_F_S;
case 31: return HD_F_S_SAMECENTER;
case 32: return HD_F_P;
case 33: return HD_F_P_SAMECENTER;
case 34: return HD_F_D;
case 35: return HD_F_D_SAMECENTER;
case 36: return HD_F_F;
case 37: return HD_F_F_SAMECENTER;
case 40: return HD_G_S;
case 41: return HD_G_S_SAMECENTER;
case 42: return HD_G_P;
case 43: return HD_G_P_SAMECENTER;
case 44: return HD_G_D;
case 45: return HD_G_D_SAMECENTER;
case 46: return HD_G_F;
case 47: return HD_G_F_SAMECENTER;
case 48: return HD_G_G;
case 49: return HD_G_G_SAMECENTER;
default:return nullptr;}}
int hrr_derivative_base_size(int a,int b,bool same) {
switch((a*5+b)*2+same) {
case 0: return 8;
case 1: return 7;
case 10: return 19;
case 11: return 16;
case 12: return 45;
case 13: return 29;
case 20: return 35;
case 21: return 29;
case 22: return 75;
case 23: return 46;
case 24: return 132;
case 25: return 67;
case 30: return 56;
case 31: return 46;
case 32: return 113;
case 33: return 67;
case 34: return 190;
case 35: return 92;
case 36: return 290;
case 37: return 121;
case 40: return 82;
case 41: return 67;
case 42: return 159;
case 43: return 92;
case 44: return 259;
case 45: return 121;
case 46: return 385;
case 47: return 154;
case 48: return 540;
case 49: return 191;
default:return 0;}}
}

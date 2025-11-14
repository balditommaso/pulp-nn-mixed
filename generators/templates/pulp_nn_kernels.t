

%if PULPNNEXT == 'XpulpV2':
#ifndef __PULPNN_KERNELS__
#define __PULPNN_KERNELS__
%elif PULPNNEXT == 'XpulpNN':
#ifndef __XPULPNN_KERNELS__
#define __XPULPNN_KERNELS__
%elif PULPNNEXT == 'XpulpNN-mixed':
#ifndef __XPULPNN-MIXED_KERNELS__
#define __XPULPNN-MIXED_KERNELS__
%endif

${PULPNNAPI}

#endif
//#include <string>
//#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include "warp_triangle_double.h"
#include "multiple_os_thread.h"


double getintensity_mindex2(int x, int y, int sizx, int sizy, double *I) {
    //return I[y*sizx+x];
	return I[x*sizy+y];
}

/* Get an pixel from an image, if outside image, black or nearest pixel */
double getcolor_mindex2(int x, int y, int sizx, int sizy, double *I, int rgb) {
    return I[rgb*sizy*sizx+y*sizx+x];
}

double interpolate_2d_cubic_gray(double Tlocalx, double Tlocaly, int *Isize, double *Iin) {
    /* Floor of coordinate */
    double fTlocalx, fTlocaly;
    /* Zero neighbor */
    int xBas0, yBas0;
    /* The location in between the pixels 0..1 */
    double tx, ty;
    /* Neighbor loccations */
    int xn[4], yn[4];
    
    /* The vectors */
    double vector_tx[4] , vector_ty[4];
    double vector_qx[4], vector_qy[4];
    /* Interpolated Intensity; */
    double Ipixel=0, Ipixelx=0;
    /* Temporary value boundary */
    int b;
    /* Loop variable */
    int i;
    
    /* Determine of the zero neighbor */
    fTlocalx = floor(Tlocalx); fTlocaly = floor(Tlocaly);
    xBas0=(int) fTlocalx; yBas0=(int) fTlocaly;
    
    /* Determine the location in between the pixels 0..1 */
    tx=Tlocalx-fTlocalx; ty=Tlocaly-fTlocaly;
    
    /* Determine the t vectors */
    vector_tx[0]= 0.5; vector_tx[1]= 0.5*tx; vector_tx[2]= 0.5*pow2(tx); vector_tx[3]= 0.5*pow3(tx);
    vector_ty[0]= 0.5; vector_ty[1]= 0.5*ty; vector_ty[2]= 0.5*pow2(ty); vector_ty[3]= 0.5*pow3(ty);
    
    /* t vector multiplied with 4x4 bicubic kernel gives the to q vectors */
    vector_qx[0]= -1.0*vector_tx[1]+2.0*vector_tx[2]-1.0*vector_tx[3];
    vector_qx[1]= 2.0*vector_tx[0]-5.0*vector_tx[2]+3.0*vector_tx[3];
    vector_qx[2]= 1.0*vector_tx[1]+4.0*vector_tx[2]-3.0*vector_tx[3];
    vector_qx[3]= -1.0*vector_tx[2]+1.0*vector_tx[3];
    vector_qy[0]= -1.0*vector_ty[1]+2.0*vector_ty[2]-1.0*vector_ty[3];
    vector_qy[1]= 2.0*vector_ty[0]-5.0*vector_ty[2]+3.0*vector_ty[3];
    vector_qy[2]= 1.0*vector_ty[1]+4.0*vector_ty[2]-3.0*vector_ty[3];
    vector_qy[3]= -1.0*vector_ty[2]+1.0*vector_ty[3];
    
    /* Determine 1D neighbour coordinates */
    xn[0]=xBas0-1; xn[1]=xBas0; xn[2]=xBas0+1; xn[3]=xBas0+2;
    yn[0]=yBas0-1; yn[1]=yBas0; yn[2]=yBas0+1; yn[3]=yBas0+2;
    
    /* Clamp to image boundary if outside image */
    if(xn[0]<0) { xn[0]=0;if(xn[1]<0) { xn[1]=0;if(xn[2]<0) { xn[2]=0; if(xn[3]<0) { xn[3]=0; }}}}
    if(yn[0]<0) { yn[0]=0;if(yn[1]<0) { yn[1]=0;if(yn[2]<0) { yn[2]=0; if(yn[3]<0) { yn[3]=0; }}}}
    b=Isize[0]-1;
    if(xn[3]>b) { xn[3]=b;if(xn[2]>b) { xn[2]=b;if(xn[1]>b) { xn[1]=b; if(xn[0]>b) { xn[0]=b; }}}}
    b=Isize[1]-1;
    if(yn[3]>b) { yn[3]=b;if(yn[2]>b) { yn[2]=b;if(yn[1]>b) { yn[1]=b; if(yn[0]>b) { yn[0]=b; }}}}
    
    /* First do interpolation in the x direction followed by interpolation in the y direction */
    for(i=0; i<4; i++) {
        Ipixelx =vector_qx[0]*getintensity_mindex2(xn[0], yn[i], Isize[0], Isize[1], Iin);
        Ipixelx+=vector_qx[1]*getintensity_mindex2(xn[1], yn[i], Isize[0], Isize[1], Iin);
        Ipixelx+=vector_qx[2]*getintensity_mindex2(xn[2], yn[i], Isize[0], Isize[1], Iin);
        Ipixelx+=vector_qx[3]*getintensity_mindex2(xn[3], yn[i], Isize[0], Isize[1], Iin);
        Ipixel+= vector_qy[i]*Ipixelx;
    }
    return Ipixel;
}

double interpolate_2d_cubic_gray_black(double Tlocalx, double Tlocaly, int *Isize, double *Iin) {
    /* Floor of coordinate */
    double fTlocalx, fTlocaly;
    /* Zero neighbor */
    int xBas0, yBas0;
    /* The location in between the pixels 0..1 */
    double tx, ty;
    /* Neighbor loccations */
    int xn[4], yn[4];
    
    /* The vectors */
    double vector_tx[4], vector_ty[4];
    double vector_qx[4], vector_qy[4];
    /* Interpolated Intensity; */
    double Ipixel=0, Ipixelx=0;
    /* Loop variable */
    int i;
    
    /* Determine of the zero neighbor */
    fTlocalx = floor(Tlocalx); fTlocaly = floor(Tlocaly);
    xBas0=(int) fTlocalx; yBas0=(int) fTlocaly;
    
    /* Determine the location in between the pixels 0..1 */
    tx=Tlocalx-fTlocalx; ty=Tlocaly-fTlocaly;
    
    /* Determine the t vectors */
    vector_tx[0]= 0.5; vector_tx[1]= 0.5*tx; vector_tx[2]= 0.5*pow2(tx); vector_tx[3]= 0.5*pow3(tx);
    vector_ty[0]= 0.5; vector_ty[1]= 0.5*ty; vector_ty[2]= 0.5*pow2(ty); vector_ty[3]= 0.5*pow3(ty);
    
    /* t vector multiplied with 4x4 bicubic kernel gives the to q vectors */
    vector_qx[0]= -1.0*vector_tx[1]+2.0*vector_tx[2]-1.0*vector_tx[3];
    vector_qx[1]= 2.0*vector_tx[0]-5.0*vector_tx[2]+3.0*vector_tx[3];
    vector_qx[2]= 1.0*vector_tx[1]+4.0*vector_tx[2]-3.0*vector_tx[3];
    vector_qx[3]= -1.0*vector_tx[2]+1.0*vector_tx[3];
    vector_qy[0]= -1.0*vector_ty[1]+2.0*vector_ty[2]-1.0*vector_ty[3];
    vector_qy[1]= 2.0*vector_ty[0]-5.0*vector_ty[2]+3.0*vector_ty[3];
    vector_qy[2]= 1.0*vector_ty[1]+4.0*vector_ty[2]-3.0*vector_ty[3];
    vector_qy[3]= -1.0*vector_ty[2]+1.0*vector_ty[3];
    
    /* Determine 1D neighbour coordinates */
    xn[0]=xBas0-1; xn[1]=xBas0; xn[2]=xBas0+1; xn[3]=xBas0+2;
    yn[0]=yBas0-1; yn[1]=yBas0; yn[2]=yBas0+1; yn[3]=yBas0+2;
    
    /* First do interpolation in the x direction followed by interpolation in the y direction */
    for(i=0; i<4; i++) {
        Ipixelx=0;
        if((yn[i]>=0)&&(yn[i]<Isize[1])) {
            if((xn[0]>=0)&&(xn[0]<Isize[0])) {
                Ipixelx+=vector_qx[0]*getintensity_mindex2(xn[0], yn[i], Isize[0], Isize[1], Iin);
            }
            if((xn[1]>=0)&&(xn[1]<Isize[0])) {
                Ipixelx+=vector_qx[1]*getintensity_mindex2(xn[1], yn[i], Isize[0], Isize[1], Iin);
            }
            if((xn[2]>=0)&&(xn[2]<Isize[0])) {
                Ipixelx+=vector_qx[2]*getintensity_mindex2(xn[2], yn[i], Isize[0], Isize[1], Iin);
            }
            if((xn[3]>=0)&&(xn[3]<Isize[0])) {
                Ipixelx+=vector_qx[3]*getintensity_mindex2(xn[3], yn[i], Isize[0], Isize[1], Iin);
            }
        }
        Ipixel+= vector_qy[i]*Ipixelx;
    }
    return Ipixel;
}

double interpolate_2d_linear_gray(double Tlocalx, double Tlocaly, int *Isize, double *Iin) {
    /*  Linear interpolation variables */
    int xBas0, xBas1, yBas0, yBas1;
    double perc[4]={0, 0, 0, 0};
    double xCom, yCom, xComi, yComi;
    double color[4]={0, 0, 0, 0};
    
    /*  Rounded location  */
    double fTlocalx, fTlocaly;
    
    /* Determine the coordinates of the pixel(s) which will be come the current pixel */
    /* (using linear interpolation) */
    fTlocalx = floor(Tlocalx); fTlocaly = floor(Tlocaly);
    xBas0=(int) fTlocalx; yBas0=(int) fTlocaly;
    xBas1=xBas0+1; yBas1=yBas0+1;
    
    /* Linear interpolation constants (percentages) */
    xCom=Tlocalx-fTlocalx; yCom=Tlocaly-fTlocaly;
    xComi=(1-xCom); yComi=(1-yCom);
    perc[0]=xComi * yComi;
    perc[1]=xComi * yCom;
    perc[2]=xCom * yComi;
    perc[3]=xCom * yCom;
    
    if(xBas0<0) { xBas0=0; if(xBas1<0) { xBas1=0; }}
    if(yBas0<0) { yBas0=0; if(yBas1<0) { yBas1=0; }}
    if(xBas1>(Isize[0]-1)) { xBas1=Isize[0]-1; if(xBas0>(Isize[0]-1)) { xBas0=Isize[0]-1; }}
    if(yBas1>(Isize[1]-1)) { yBas1=Isize[1]-1; if(yBas0>(Isize[1]-1)) { yBas0=Isize[1]-1; }}
    
    color[0]=getintensity_mindex2(xBas0, yBas0, Isize[0], Isize[1], Iin);
    color[1]=getintensity_mindex2(xBas0, yBas1, Isize[0], Isize[1], Iin);
    color[2]=getintensity_mindex2(xBas1, yBas0, Isize[0], Isize[1], Iin);
    color[3]=getintensity_mindex2(xBas1, yBas1, Isize[0], Isize[1], Iin);
    return color[0]*perc[0]+color[1]*perc[1]+color[2]*perc[2]+color[3]*perc[3];
}

double interpolate_2d_linear_gray_black(double Tlocalx, double Tlocaly, int *Isize, double *Iin) {
    /*  Linear interpolation variables */
    int xBas0, xBas1, yBas0, yBas1;
    double perc[4]={0, 0, 0, 0};
    double xCom, yCom, xComi, yComi;
    double Ipixel=0;
    
    
    /*  Rounded location  */
    double fTlocalx, fTlocaly;
    
    /* Determine the coordinates of the pixel(s) which will be come the current pixel */
    /* (using linear interpolation) */
    fTlocalx = floor(Tlocalx); fTlocaly = floor(Tlocaly);
    xBas0=(int) fTlocalx; yBas0=(int) fTlocaly;
    xBas1=xBas0+1; yBas1=yBas0+1;
    
    /* Linear interpolation constants (percentages) */
    xCom=Tlocalx-fTlocalx; yCom=Tlocaly-fTlocaly;
    xComi=(1-xCom); yComi=(1-yCom);
    perc[0]=xComi * yComi; perc[1]=xComi * yCom; perc[2]=xCom * yComi; perc[3]=xCom * yCom;
    
    if((xBas0>=0)&&(xBas0<Isize[0])) {
        if((yBas0>=0)&&(yBas0<Isize[1])) {
            Ipixel+=getintensity_mindex2(xBas0, yBas0, Isize[0], Isize[1], Iin)*perc[0];
        }
        if((yBas1>=0)&&(yBas1<Isize[1])) {
            Ipixel+=getintensity_mindex2(xBas0, yBas1, Isize[0], Isize[1], Iin)*perc[1];
        }
    }
    if((xBas1>=0)&&(xBas1<Isize[0]))  {
        if((yBas0>=0)&&(yBas0<Isize[1])) {
            Ipixel+=getintensity_mindex2(xBas1, yBas0, Isize[0], Isize[1], Iin)*perc[2];
        }
        if((yBas1>=0)&&(yBas1<Isize[1])) {
            Ipixel+=getintensity_mindex2(xBas1, yBas1, Isize[0], Isize[1], Iin)*perc[3];
        }
    }
    return Ipixel;
}

double interpolate_2d_double_gray(double Tlocalx, double Tlocaly, int *Isize, double *Iin, int cubic, int black) {
    double Ipixel;
    if(cubic) {
        if(black) { Ipixel=interpolate_2d_cubic_gray_black(Tlocalx, Tlocaly, Isize, Iin); }
        else { Ipixel=interpolate_2d_cubic_gray(Tlocalx, Tlocaly, Isize, Iin); }
    }
    else {
        if(black) { Ipixel=interpolate_2d_linear_gray_black(Tlocalx, Tlocaly, Isize, Iin); }
        else { Ipixel=interpolate_2d_linear_gray(Tlocalx, Tlocaly, Isize, Iin); }
    }
    return Ipixel;
}

void interpolate_2d_cubic_color_black(double *Ipixel, double Tlocalx, double Tlocaly, int *Isize, double *Iin) {
    /* Floor of coordinate */
    double fTlocalx, fTlocaly;
    /* Zero neighbor */
    int xBas0, yBas0;
    /* The location in between the pixels 0..1 */
    double tx, ty;
    /* Neighbor loccations */
    int xn[4], yn[4];
    
    /* The vectors */
    double vector_tx[4], vector_ty[4];
    double vector_qx[4], vector_qy[4];
    double Ipixelx[3];
    /* Loop variable */
    int i;
    
    /* Determine of the zero neighbor */
    fTlocalx = floor(Tlocalx); fTlocaly = floor(Tlocaly);
    xBas0=(int) fTlocalx; yBas0=(int) fTlocaly;
    
    /* Determine the location in between the pixels 0..1 */
    tx=Tlocalx-fTlocalx; ty=Tlocaly-fTlocaly;
    
    /* Determine the t vectors */
    vector_tx[0]= 0.5; vector_tx[1]= 0.5*tx; vector_tx[2]= 0.5*pow2(tx); vector_tx[3]= 0.5*pow3(tx);
    vector_ty[0]= 0.5; vector_ty[1]= 0.5*ty; vector_ty[2]= 0.5*pow2(ty); vector_ty[3]= 0.5*pow3(ty);
    
    /* t vector multiplied with 4x4 bicubic kernel gives the to q vectors */
    vector_qx[0]= -1.0*vector_tx[1]+2.0*vector_tx[2]-1.0*vector_tx[3];
    vector_qx[1]= 2.0*vector_tx[0]-5.0*vector_tx[2]+3.0*vector_tx[3];
    vector_qx[2]= 1.0*vector_tx[1]+4.0*vector_tx[2]-3.0*vector_tx[3];
    vector_qx[3]= -1.0*vector_tx[2]+1.0*vector_tx[3];
    vector_qy[0]= -1.0*vector_ty[1]+2.0*vector_ty[2]-1.0*vector_ty[3];
    vector_qy[1]= 2.0*vector_ty[0]-5.0*vector_ty[2]+3.0*vector_ty[3];
    vector_qy[2]= 1.0*vector_ty[1]+4.0*vector_ty[2]-3.0*vector_ty[3];
    vector_qy[3]= -1.0*vector_ty[2]+1.0*vector_ty[3];
    
    /* Determine 1D neighbour coordinates */
    xn[0]=xBas0-1; xn[1]=xBas0; xn[2]=xBas0+1; xn[3]=xBas0+2;
    yn[0]=yBas0-1; yn[1]=yBas0; yn[2]=yBas0+1; yn[3]=yBas0+2;
    
    /* First do interpolation in the x direction followed by interpolation in the y direction */
    Ipixel[0]=0; Ipixel[1]=0; Ipixel[2]=0;
    for(i=0; i<4; i++) {
        Ipixelx[0]=0; Ipixelx[1]=0; Ipixelx[2]=0;
        if((yn[i]>=0)&&(yn[i]<Isize[1])) {
            if((xn[0]>=0)&&(xn[0]<Isize[0])) {
                Ipixelx[0]+=vector_qx[0]*getcolor_mindex2(xn[0], yn[i], Isize[0], Isize[1], Iin, 0);
                Ipixelx[1]+=vector_qx[0]*getcolor_mindex2(xn[0], yn[i], Isize[0], Isize[1], Iin, 1);
                Ipixelx[2]+=vector_qx[0]*getcolor_mindex2(xn[0], yn[i], Isize[0], Isize[1], Iin, 2);
            }
            if((xn[1]>=0)&&(xn[1]<Isize[0])) {
                Ipixelx[0]+=vector_qx[1]*getcolor_mindex2(xn[1], yn[i], Isize[0], Isize[1], Iin, 0);
                Ipixelx[1]+=vector_qx[1]*getcolor_mindex2(xn[1], yn[i], Isize[0], Isize[1], Iin, 1);
                Ipixelx[2]+=vector_qx[1]*getcolor_mindex2(xn[1], yn[i], Isize[0], Isize[1], Iin, 2);
            }
            if((xn[2]>=0)&&(xn[2]<Isize[0])) {
                Ipixelx[0]+=vector_qx[2]*getcolor_mindex2(xn[2], yn[i], Isize[0], Isize[1], Iin, 0);
                Ipixelx[1]+=vector_qx[2]*getcolor_mindex2(xn[2], yn[i], Isize[0], Isize[1], Iin, 1);
                Ipixelx[2]+=vector_qx[2]*getcolor_mindex2(xn[2], yn[i], Isize[0], Isize[1], Iin, 2);
            }
            if((xn[3]>=0)&&(xn[3]<Isize[0])) {
                Ipixelx[0]+=vector_qx[3]*getcolor_mindex2(xn[3], yn[i], Isize[0], Isize[1], Iin, 0);
                Ipixelx[1]+=vector_qx[3]*getcolor_mindex2(xn[3], yn[i], Isize[0], Isize[1], Iin, 1);
                Ipixelx[2]+=vector_qx[3]*getcolor_mindex2(xn[3], yn[i], Isize[0], Isize[1], Iin, 2);
            }
        }
        Ipixel[0]+= vector_qy[i]*Ipixelx[0];
        Ipixel[1]+= vector_qy[i]*Ipixelx[1];
        Ipixel[2]+= vector_qy[i]*Ipixelx[2];
    }
}

void interpolate_2d_cubic_color(double *Ipixel, double Tlocalx, double Tlocaly, int *Isize, double *Iin) {
    /* Floor of coordinate */
    double fTlocalx, fTlocaly;
    /* Zero neighbor */
    int xBas0, yBas0;
    /* The location in between the pixels 0..1 */
    double tx, ty;
    /* Neighbor loccations */
    int xn[4], yn[4];
    
    /* The vectors */
    double vector_tx[4], vector_ty[4];
    double vector_qx[4], vector_qy[4];
    double Ipixelx;
    /* Temporary value boundary */
    int b;
    /* Loop variable */
    int i, rgb;
    
    /* Determine of the zero neighbor */
    fTlocalx = floor(Tlocalx); fTlocaly = floor(Tlocaly);
    xBas0=(int) fTlocalx; yBas0=(int) fTlocaly;
    
    /* Determine the location in between the pixels 0..1 */
    tx=Tlocalx-fTlocalx; ty=Tlocaly-fTlocaly;
    
    /* Determine the t vectors */
    vector_tx[0]= 0.5; vector_tx[1]= 0.5*tx; vector_tx[2]= 0.5*pow2(tx); vector_tx[3]= 0.5*pow3(tx);
    vector_ty[0]= 0.5; vector_ty[1]= 0.5*ty; vector_ty[2]= 0.5*pow2(ty); vector_ty[3]= 0.5*pow3(ty);
    
    /* t vector multiplied with 4x4 bicubic kernel gives the to q vectors */
    vector_qx[0]= -1.0*vector_tx[1]+2.0*vector_tx[2]-1.0*vector_tx[3];
    vector_qx[1]= 2.0*vector_tx[0]-5.0*vector_tx[2]+3.0*vector_tx[3];
    vector_qx[2]= 1.0*vector_tx[1]+4.0*vector_tx[2]-3.0*vector_tx[3];
    vector_qx[3]= -1.0*vector_tx[2]+1.0*vector_tx[3];
    vector_qy[0]= -1.0*vector_ty[1]+2.0*vector_ty[2]-1.0*vector_ty[3];
    vector_qy[1]= 2.0*vector_ty[0]-5.0*vector_ty[2]+3.0*vector_ty[3];
    vector_qy[2]= 1.0*vector_ty[1]+4.0*vector_ty[2]-3.0*vector_ty[3];
    vector_qy[3]= -1.0*vector_ty[2]+1.0*vector_ty[3];
    
    /* Determine 1D neighbour coordinates */
    xn[0]=xBas0-1; xn[1]=xBas0; xn[2]=xBas0+1; xn[3]=xBas0+2;
    yn[0]=yBas0-1; yn[1]=yBas0; yn[2]=yBas0+1; yn[3]=yBas0+2;
    
    /* Clamp to image boundary if outside image */
    if(xn[0]<0) { xn[0]=0;if(xn[1]<0) { xn[1]=0;if(xn[2]<0) { xn[2]=0; if(xn[3]<0) { xn[3]=0; }}}}
    if(yn[0]<0) { yn[0]=0;if(yn[1]<0) { yn[1]=0;if(yn[2]<0) { yn[2]=0; if(yn[3]<0) { yn[3]=0; }}}}
    b=Isize[0]-1;
    if(xn[3]>b) { xn[3]=b;if(xn[2]>b) { xn[2]=b;if(xn[1]>b) { xn[1]=b; if(xn[0]>b) { xn[0]=b; }}}}
    b=Isize[1]-1;
    if(yn[3]>b) { yn[3]=b;if(yn[2]>b) { yn[2]=b;if(yn[1]>b) { yn[1]=b; if(yn[0]>b) { yn[0]=b; }}}}
    
    /* First do interpolation in the x direction followed by interpolation in the y direction */
    for (rgb=0; rgb<3; rgb++) {
        Ipixel[rgb]=0;
        for(i=0; i<4; i++) {
            Ipixelx =vector_qx[0]*getcolor_mindex2(xn[0], yn[i], Isize[0], Isize[1], Iin, rgb);
            Ipixelx+=vector_qx[1]*getcolor_mindex2(xn[1], yn[i], Isize[0], Isize[1], Iin, rgb);
            Ipixelx+=vector_qx[2]*getcolor_mindex2(xn[2], yn[i], Isize[0], Isize[1], Iin, rgb);
            Ipixelx+=vector_qx[3]*getcolor_mindex2(xn[3], yn[i], Isize[0], Isize[1], Iin, rgb);
            Ipixel[rgb]+= vector_qy[i]*Ipixelx;
        }
    }
}

void interpolate_2d_linear_color_black(double *Ipixel, double Tlocalx, double Tlocaly, int *Isize, double *Iin) {
    /*  Linear interpolation variables */
    int xBas0, xBas1, yBas0, yBas1;
    double perc[4]={0, 0, 0, 0};
    double xCom, yCom, xComi, yComi;
    
    /* Loop variable  */
    int rgb;
    
    /*  Rounded location  */
    double fTlocalx, fTlocaly;
    
    Ipixel[0]=0; Ipixel[1]=0; Ipixel[2]=0;
    /* Determine the coordinates of the pixel(s) which will be come the current pixel */
    /* (using linear interpolation) */
    fTlocalx = floor(Tlocalx); fTlocaly = floor(Tlocaly);
    xBas0=(int) fTlocalx; yBas0=(int) fTlocaly;
    xBas1=xBas0+1; yBas1=yBas0+1;
    
    /* Linear interpolation constants (percentages) */
    xCom=Tlocalx-fTlocalx; yCom=Tlocaly-fTlocaly;
    xComi=(1-xCom); yComi=(1-yCom);
    perc[0]=xComi * yComi; perc[1]=xComi * yCom; perc[2]=xCom * yComi; perc[3]=xCom * yCom;
    
    if((xBas0>=0)&&(xBas0<Isize[0])) {
        if((yBas0>=0)&&(yBas0<Isize[1])) {
            for (rgb=0; rgb<3; rgb++) {
                Ipixel[rgb]+=getcolor_mindex2(xBas0, yBas0, Isize[0], Isize[1], Iin, rgb)*perc[0];
            }
        }
        if((yBas1>=0)&&(yBas1<Isize[1])) {
            for (rgb=0; rgb<3; rgb++) {
                Ipixel[rgb]+=getcolor_mindex2(xBas0, yBas1, Isize[0], Isize[1], Iin, rgb)*perc[1];
            }
        }
    }
    if((xBas1>=0)&&(xBas1<Isize[0]))  {
        if((yBas0>=0)&&(yBas0<Isize[1])) {
            for (rgb=0; rgb<3; rgb++) {
                Ipixel[rgb]+=getcolor_mindex2(xBas1, yBas0, Isize[0], Isize[1], Iin, rgb)*perc[2];
            }
        }
        if((yBas1>=0)&&(yBas1<Isize[1])) {
            for (rgb=0; rgb<3; rgb++) {
                Ipixel[rgb]+=getcolor_mindex2(xBas1, yBas1, Isize[0], Isize[1], Iin, rgb)*perc[3];
            }
        }
    }
}

void interpolate_2d_linear_color(double *Ipixel, double Tlocalx, double Tlocaly, int *Isize, double *Iin) {
    /*  Linear interpolation variables */
    int xBas0, xBas1, yBas0, yBas1;
    double perc[4]={0, 0, 0, 0};
    double xCom, yCom, xComi, yComi;
    double color[4]={0, 0, 0, 0};
    
    /* Loop variable  */
    int rgb;
    
    /*  Rounded location  */
    double fTlocalx, fTlocaly;
    
    /* Determine the coordinates of the pixel(s) which will be come the current pixel */
    /* (using linear interpolation) */
    fTlocalx = floor(Tlocalx); fTlocaly = floor(Tlocaly);
    xBas0=(int) fTlocalx; yBas0=(int) fTlocaly;
    xBas1=xBas0+1; yBas1=yBas0+1;
    
    /* Linear interpolation constants (percentages) */
    xCom=Tlocalx-fTlocalx; yCom=Tlocaly-fTlocaly;
    xComi=(1-xCom); yComi=(1-yCom);
    perc[0]=xComi * yComi;
    perc[1]=xComi * yCom;
    perc[2]=xCom * yComi;
    perc[3]=xCom * yCom;
    
    if(xBas0<0) { xBas0=0; if(xBas1<0) { xBas1=0; }}
    if(yBas0<0) { yBas0=0; if(yBas1<0) { yBas1=0; }}
    if(xBas1>(Isize[0]-1)) { xBas1=Isize[0]-1; if(xBas0>(Isize[0]-1)) { xBas0=Isize[0]-1; }}
    if(yBas1>(Isize[1]-1)) { yBas1=Isize[1]-1; if(yBas0>(Isize[1]-1)) { yBas0=Isize[1]-1; }}
    
    for (rgb=0; rgb<3; rgb++) {
        color[0]=getcolor_mindex2(xBas0, yBas0, Isize[0], Isize[1], Iin, rgb);
        color[1]=getcolor_mindex2(xBas0, yBas1, Isize[0], Isize[1], Iin, rgb);
        color[2]=getcolor_mindex2(xBas1, yBas0, Isize[0], Isize[1], Iin, rgb);
        color[3]=getcolor_mindex2(xBas1, yBas1, Isize[0], Isize[1], Iin, rgb);
        Ipixel[rgb]=color[0]*perc[0]+color[1]*perc[1]+color[2]*perc[2]+color[3]*perc[3];
    }
}

void interpolate_2d_double_color(double *Ipixel, double Tlocalx, double Tlocaly, int *Isize, double *Iin, int cubic, int black) {
    if(cubic) {
        if(black) { interpolate_2d_cubic_color_black(Ipixel, Tlocalx, Tlocaly, Isize, Iin);}
        else { interpolate_2d_cubic_color(Ipixel, Tlocalx, Tlocaly, Isize, Iin);}
    }
    else {
        if(black) { interpolate_2d_linear_color_black(Ipixel, Tlocalx, Tlocaly, Isize, Iin);}
        else { interpolate_2d_linear_color(Ipixel, Tlocalx, Tlocaly, Isize, Iin);}		
    }
}

void warp_triangle_double2(double *Iin, int* size_Iin, double *Iout, int* size_Iout, 
						  double *XY, int* size_XY, double *UV, int* size_UV, double *TRI, int* size_TRI)
{
    //double *Iout, *Iin, *XY, *UV, *TRI, *SizeO;
    
    //const mwSize *Idims;
	//const mwSize *Tdims;
    //const mwSize *Vdims;
    
    int Isize[3] = {1,1,1};
    int Osize[3] = {1,1,1};
    int Tsize[2] = {1,1};
    int Vsize[2] = {1,1};
    int Oslice;
    /* loop variable */
    int q, i, j, rgb;
    
    /* Current Vertices indices*/
    int t0, t1, t2;
    
    /* Current voxel/pixel */
    double Ipixel[3]={0,0,0};
    
    /* Bounding box polygon */
    int boundmin[2];
    int boundmax[2];
    
    /* Vertices */
    double p0[2],p1[2], p2[2];
    double x0[2],x1[2], x2[2];
    
    /* Barycentric variables */
    double f12, f20, f01;
    double g12[2],g20[2],g01[2];
    double c12,c20,c01;
    double posuv[2];
    
    /* Interpolation percentages */
    double Lambda[3];
    double Lambdat[3];
    
    /* Boundary values */
    const double mval=-0.0001;
    const double bval=1.0001;
    
    /* function J=warp_triangle_double(I,xy,uv,tri,ImageSize) */
    //Iin=mxGetPr(prhs[0]);
    //XY=mxGetPr(prhs[1]);
    //UV=mxGetPr(prhs[2]);
    //TRI=mxGetPr(prhs[3]);
    //SizeO=mxGetPr(prhs[4]);
    
    /* Input Image size */
    //Idims = mxGetDimensions(prhs[0]);
	//Idims = size_Iin;Isize[0] = Idims[0];Isize[1] = Idims[1];
    Isize[0] = size_Iin[0];Isize[1] = size_Iin[1];

    /* Input Number of Polygons */
    //Tdims = mxGetDimensions(prhs[3]);Tsize[0] = Tdims[0];Tsize[1] = Tdims[1];
    Tsize[0] = size_TRI[0];Tsize[1] = size_TRI[1];

    /* Input Number of Polygons */
    //Vdims = mxGetDimensions(prhs[1]);Vsize[0] = Vdims[0];Vsize[1] = Vdims[1];
    Vsize[0] = size_XY[0];Vsize[1] = size_XY[1];
    
    //if(mxGetNumberOfDimensions(prhs[0])>2) { Isize[2] = 3; }
	if(size_Iin[2]>1) { Isize[2] = 3; }

    /* Output Image size */
    Osize[0]=(int)size_Iout[0];
    Osize[1]=(int)size_Iout[1];
    Osize[2]=Isize[2];
    Oslice=Osize[0]*Osize[1];
    
    /* Create empty array for output */
    //plhs[0] = mxCreateNumericArray(3, Osize, mxDOUBLE_CLASS, mxREAL);
    //Iout=mxGetPr(plhs[0]);
    
	//if(nrhs<6){Iout=memset(Iout,0,Osize[0]*Osize[1]*Osize[2]*sizeof(double));}
    //else{Iout=memcpy(Iout,mxGetData(prhs[5]),Osize[0]*Osize[1]*Osize[2]*sizeof(double));}
    
    for(q=0; q<Tsize[0]; q++)
    {
        t0=(int)TRI[q];
        t1=(int)TRI[q+Tsize[0]];
        t2=(int)TRI[q+Tsize[0]*2];
        
        /* Vertices */
        p0[0]=UV[t0]; p0[1]=UV[t0+Vsize[0]];
        p1[0]=UV[t1]; p1[1]=UV[t1+Vsize[0]];
        p2[0]=UV[t2]; p2[1]=UV[t2+Vsize[0]];
        
        /* Vertices2*/
        x0[0]=XY[t0]; x0[1]=XY[t0+Vsize[0]];
        x1[0]=XY[t1]; x1[1]=XY[t1+Vsize[0]];
        x2[0]=XY[t2]; x2[1]=XY[t2+Vsize[0]];
        
        /*  Get bounding box (ROI) */
        boundmin[0]=(int)floor(min(min(p0[0],p1[0]),p2[0]));
        boundmin[1]=(int)floor(min(min(p0[1],p1[1]),p2[1]));
        
        boundmax[0]=(int) ceil(max(max(p0[0],p1[0]),p2[0]));
        boundmax[1]=(int) ceil(max(max(p0[1],p1[1]),p2[1]));
        
        boundmin[0]=max(boundmin[0],0);
        boundmin[1]=max(boundmin[1],0);
        
        boundmax[0]=min(boundmax[0],Osize[0]-1);
        boundmax[1]=min(boundmax[1],Osize[1]-1);
        
        /* Normalization factors */
        f12 = ( p1[1] - p2[1] ) * p0[0]  + (p2[0] - p1[0] ) * p0[1] + p1[0] * p2[1] - p2[0] *p1[1];
        f20 = ( p2[1] - p0[1] ) * p1[0]  + (p0[0] - p2[0] ) * p1[1] + p2[0] * p0[1] - p0[0] *p2[1];
        f01 = ( p0[1] - p1[1] ) * p2[0]  + (p1[0] - p0[0] ) * p2[1] + p0[0] * p1[1] - p1[0] *p0[1];
        
        /* Lambda Gradient */
        g12[0]=( p1[1] - p2[1] )/f12; g12[1] = (p2[0] - p1[0] )/f12;
        g20[0]=( p2[1] - p0[1] )/f20; g20[1] = (p0[0] - p2[0] )/f20;
        g01[0]=( p0[1] - p1[1] )/f01; g01[1] = (p1[0] - p0[0] )/f01;
        
        /* Center compensation */
        c12 = (p1[0] * p2[1] - p2[0] *p1[1])/f12;
        c20 = (p2[0] * p0[1] - p0[0] *p2[1])/f20;
        c01 = (p0[0] * p1[1] - p1[0] *p0[1])/f01;
        
        Lambdat[0]=g12[1]*boundmin[1]+c12+g12[0]*boundmin[0];;
        Lambdat[1]=g20[1]*boundmin[1]+c20+g20[0]*boundmin[0];;
        Lambdat[2]=g01[1]*boundmin[1]+c01+g01[0]*boundmin[0];;
        for(j=boundmin[1]; j<=boundmax[1]; j++)
        {
            Lambda[0] = Lambdat[0];
            Lambda[1] = Lambdat[1];
            Lambda[2] = Lambdat[2];
            for(i=boundmin[0]; i<=boundmax[0]; i++)
            {
                /* Check if voxel is inside the triangle */
                if((Lambda[0]>mval)&&(Lambda[0]<bval)&&(Lambda[1]>mval)&&(Lambda[1]<bval)&&(Lambda[2]>mval)&&(Lambda[2]<bval))
                {
                    posuv[0]=Lambda[0]*x0[0]+Lambda[1]*x1[0]+Lambda[2]*x2[0];
                    posuv[1]=Lambda[0]*x0[1]+Lambda[1]*x1[1]+Lambda[2]*x2[1];
                    
                    if(Osize[2]>1)
                    {
                        interpolate_2d_double_color(Ipixel,posuv[0], posuv[1], Isize, Iin, true, false);
                    }
                    else
                    {
                        Ipixel[0]=interpolate_2d_double_gray(posuv[0], posuv[1], Isize, Iin, true, false);
                    }
                    
                    for(rgb=0;rgb<Osize[2];rgb++)
                    {
                        Iout[i+j*Osize[0]+rgb*Oslice]=Ipixel[rgb];
                    }
                }
                Lambda[0] += g12[0];
                Lambda[1] += g20[0];
                Lambda[2] += g01[0];
            }
            Lambdat[0] += g12[1];
            Lambdat[1] += g20[1];
            Lambdat[2] += g01[1];
        }
    }
}

void warp_triangle_double(double *Iin, int* size_Iin, double *Iout, int* size_Iout, 
						  double *XY, int* size_XY, double *UV, int* size_UV, double *TRI, int* size_TRI)
{
    //double *Iout, *Iin, *XY, *UV, *TRI, *SizeO;
    
    //const mwSize *Idims;
	//const mwSize *Tdims;
    //const mwSize *Vdims;
    
    int Isize[3] = {1,1,1};
    int Osize[3] = {1,1,1};
    int Tsize[2] = {1,1};
    int Vsize[2] = {1,1};
    int Oslice;
    /* loop variable */
    int q, i, j, rgb;
    
    /* Current Vertices indices*/
    int t0, t1, t2;
    
    /* Current voxel/pixel */
    double Ipixel[3]={0,0,0};
    
    /* Bounding box polygon */
    int boundmin[2];
    int boundmax[2];
    
    /* Vertices */
    double p0[2],p1[2], p2[2];
    double x0[2],x1[2], x2[2];
    
    /* Barycentric variables */
    double f12, f20, f01;
    double g12[2],g20[2],g01[2];
    double c12,c20,c01;
    double posuv[2];
    
    /* Interpolation percentages */
    double Lambda[3];
    double Lambdat[3];
    
    /* Boundary values */
    const double mval=-0.0001;
    const double bval=1.0001;
    
    /* function J=warp_triangle_double(I,xy,uv,tri,ImageSize) */
    //Iin=mxGetPr(prhs[0]);
    //XY=mxGetPr(prhs[1]);
    //UV=mxGetPr(prhs[2]);
    //TRI=mxGetPr(prhs[3]);
    //SizeO=mxGetPr(prhs[4]);
    
    /* Input Image size */
    //Idims = mxGetDimensions(prhs[0]);
	//Idims = size_Iin;Isize[0] = Idims[0];Isize[1] = Idims[1];
    Isize[0] = size_Iin[0];Isize[1] = size_Iin[1];

    /* Input Number of Polygons */
    //Tdims = mxGetDimensions(prhs[3]);Tsize[0] = Tdims[0];Tsize[1] = Tdims[1];
    Tsize[0] = size_TRI[0];Tsize[1] = size_TRI[1];

    /* Input Number of Polygons */
    //Vdims = mxGetDimensions(prhs[1]);Vsize[0] = Vdims[0];Vsize[1] = Vdims[1];
    Vsize[0] = size_XY[0];Vsize[1] = size_XY[1];
    
    //if(mxGetNumberOfDimensions(prhs[0])>2) { Isize[2] = 3; }
	if(size_Iin[2]>1) { Isize[2] = 3; }

    /* Output Image size */
    Osize[0]=(int)size_Iout[0];
    Osize[1]=(int)size_Iout[1];
    Osize[2]=Isize[2];
    Oslice=Osize[0]*Osize[1];
    
    /* Create empty array for output */
    //plhs[0] = mxCreateNumericArray(3, Osize, mxDOUBLE_CLASS, mxREAL);
    //Iout=mxGetPr(plhs[0]);
    
	//if(nrhs<6){Iout=memset(Iout,0,Osize[0]*Osize[1]*Osize[2]*sizeof(double));}
    //else{Iout=memcpy(Iout,mxGetData(prhs[5]),Osize[0]*Osize[1]*Osize[2]*sizeof(double));}
    
    for(q=0; q<Tsize[0]; q++)
    {
        t0=(int)TRI[q*Tsize[1]];
        t1=(int)TRI[q*Tsize[1]+1];
        t2=(int)TRI[q*Tsize[1]+2];
        
        /* Vertices */
        p0[1]=UV[t0*Vsize[1]]; p0[0]=UV[t0*Vsize[1]+1];
        p1[1]=UV[t1*Vsize[1]]; p1[0]=UV[t1*Vsize[1]+1];
        p2[1]=UV[t2*Vsize[1]]; p2[0]=UV[t2*Vsize[1]+1];
        
        /* Vertices2*/
        x0[1]=XY[t0*Vsize[1]]; x0[0]=XY[t0*Vsize[1]+1];
        x1[1]=XY[t1*Vsize[1]]; x1[0]=XY[t1*Vsize[1]+1];
        x2[1]=XY[t2*Vsize[1]]; x2[0]=XY[t2*Vsize[1]+1];
        
        /*  Get bounding box (ROI) */
        boundmin[0]=(int)floor(min(min(p0[0],p1[0]),p2[0]));
        boundmin[1]=(int)floor(min(min(p0[1],p1[1]),p2[1]));
        
        boundmax[0]=(int) ceil(max(max(p0[0],p1[0]),p2[0]));
        boundmax[1]=(int) ceil(max(max(p0[1],p1[1]),p2[1]));
        
        boundmin[0]=max(boundmin[0],0);
        boundmin[1]=max(boundmin[1],0);
        
        boundmax[0]=min(boundmax[0],Osize[0]-1);
        boundmax[1]=min(boundmax[1],Osize[1]-1);
        
        /* Normalization factors */
        f12 = ( p1[1] - p2[1] ) * p0[0]  + (p2[0] - p1[0] ) * p0[1] + p1[0] * p2[1] - p2[0] *p1[1];
        f20 = ( p2[1] - p0[1] ) * p1[0]  + (p0[0] - p2[0] ) * p1[1] + p2[0] * p0[1] - p0[0] *p2[1];
        f01 = ( p0[1] - p1[1] ) * p2[0]  + (p1[0] - p0[0] ) * p2[1] + p0[0] * p1[1] - p1[0] *p0[1];
        
        /* Lambda Gradient */
        g12[0]=( p1[1] - p2[1] )/f12; g12[1] = (p2[0] - p1[0] )/f12;
        g20[0]=( p2[1] - p0[1] )/f20; g20[1] = (p0[0] - p2[0] )/f20;
        g01[0]=( p0[1] - p1[1] )/f01; g01[1] = (p1[0] - p0[0] )/f01;
        
        /* Center compensation */
        c12 = (p1[0] * p2[1] - p2[0] *p1[1])/f12;
        c20 = (p2[0] * p0[1] - p0[0] *p2[1])/f20;
        c01 = (p0[0] * p1[1] - p1[0] *p0[1])/f01;
        
        Lambdat[0]=g12[1]*boundmin[1]+c12+g12[0]*boundmin[0];;
        Lambdat[1]=g20[1]*boundmin[1]+c20+g20[0]*boundmin[0];;
        Lambdat[2]=g01[1]*boundmin[1]+c01+g01[0]*boundmin[0];;
        for(j=boundmin[1]; j<=boundmax[1]; j++)
        {
            Lambda[0] = Lambdat[0];
            Lambda[1] = Lambdat[1];
            Lambda[2] = Lambdat[2];
            for(i=boundmin[0]; i<=boundmax[0]; i++)
            {
                /* Check if voxel is inside the triangle */
                if((Lambda[0]>mval)&&(Lambda[0]<bval)&&(Lambda[1]>mval)&&(Lambda[1]<bval)&&(Lambda[2]>mval)&&(Lambda[2]<bval))
                {
                    posuv[0]=Lambda[0]*x0[0]+Lambda[1]*x1[0]+Lambda[2]*x2[0];
                    posuv[1]=Lambda[0]*x0[1]+Lambda[1]*x1[1]+Lambda[2]*x2[1];
                    
                    if(Osize[2]>1)
                    {
                        interpolate_2d_double_color(Ipixel, posuv[0], posuv[1], Isize, Iin, true, false);
                    }
                    else
                    {
                        Ipixel[0]=interpolate_2d_double_gray(posuv[0], posuv[1], Isize, Iin, true, false);
                    }
                    
                    for(rgb=0;rgb<Osize[2];rgb++)
                    {
                        Iout[j+i*Osize[1]+rgb*Oslice]=Ipixel[rgb];
                    }
                }
                Lambda[0] += g12[0];
                Lambda[1] += g20[0];
                Lambda[2] += g01[0];
            }
            Lambdat[0] += g12[1];
            Lambdat[1] += g20[1];
            Lambdat[2] += g01[1];
        }
    }
}
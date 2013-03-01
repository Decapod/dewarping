#include <stdlib.h>
#include <stdio.h>
#include <math.h>

extern "C" {

    double eval_poly(double* poly,int order,double x) {
        double val = 0.0;
        for(int i=0;i<order;i++) {
            val += poly[i]*pow(x,i);
        }
        return val;
    }

    unsigned int col_at(unsigned int* img,int h,int w,double x,double y,double* poly_sep,int poly_sep_order,int direction) {
        if(x<0||y<0)
            return 0;
        int xh = ceil(x);
        if(xh>=w)
            return 0;
        int yh = ceil(y);
        if(yh>=h)
            return 0;
        if(direction!=(x<eval_poly(poly_sep,poly_sep_order,y)))
            return 0;
        int xl = floor(x);
        int yl = floor(y);
        double xhd = xh-x;
        double yhd = yh-y;
        double xld = x-xl;
        double yld = y-yl;
        double val = double(img[yl*w+xl])*xhd*yhd+double(img[yl*w+xh])*xld*yhd \
                   + double(img[yh*w+xl])*xhd*yld+double(img[yh*w+xh])*xld*yld;
        return (unsigned int)round(val);
    }

    void calc_line_means(int* labels,double* means,int h,int w,int n_lines) {
        int* ns = (int*)malloc(sizeof(int)*n_lines);
        int* sums = (int*)malloc(sizeof(int)*n_lines);
        for(int x=0;x<w;x++) {
            for(int i=0;i<n_lines;i++) {
                ns[i] = 0;
                sums[i] = 0;
            }
            for(int y=0;y<h;y++) {
                unsigned int val = labels[y*w+x];
                if(val==0)
                    continue;
                ns[val-1]++;
                sums[val-1] += y;
            }
            for(int i=0;i<n_lines;i++) {
                if(ns[i]==0)
                    continue;
                means[i*w+x] = (double)sums[i]/(double)ns[i];
            }
        }
        free(ns);
        free(sums);
    }

}

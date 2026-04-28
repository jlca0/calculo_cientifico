#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <time.h>

using namespace std;

double h_inicial(double x0, double y0) {    
	double ci;
    ci= 1 + exp(-4*(x0*x0+y0*y0)); 
    return ci;
}
double q1_inicial(double x0, double y0) {    
    return 0;
}

double q2_inicial(double x0, double y0) {    
	return 0;
}

int main(int argc, char **argv) {
	double a,b,c,d,T,cfl,tiempo;
	int npx,npy;
	char* fichero_salida;	
	double dx,dy,dt,dtg,tg;
    double max1, max2;
	FILE *fp;
    clock_t t_ini, t_fin;
    double secs;
    t_ini = clock();
	if (argc != 11) {
		printf("Uso:\n");
		printf(argv[0]);
		printf("  a b c d T npx npy cfl c1 c2 fichero_salida  dtg\n");
		printf("a: Comienzo del intervalo, direccion x.\n");
		printf("b: Final del inervalo, direccion x.\n");
        printf("c: Comienzo del intervalo, direccion y.\n");
		printf("d: Final del inervalo, direccion y\n");
		printf("T: Tiempo total de integracion.\n");
		printf("npx: N. de particiones del intervalo [a,b]\n");
        printf("npy: N. de particiones del intervalo [c,d]\n");
		printf("cfl: Coef. estabilidad.\n");
		printf("Nombre fichero de salida.\n");
        printf("Dt guardado.\n");
		return -1;
	};
	a=atof(argv[1]);
	b=atof(argv[2]);
    c=atof(argv[3]);
	d=atof(argv[4]);
	T=atof(argv[5]);
	npx=atoi(argv[6]);
    npy=atoi(argv[7]);
	cfl=atof(argv[8]);
	fichero_salida=argv[9];
    dtg=atof(argv[10]);
	/* verificar que a>b etc...*/
	/* T>0*/
	/* 0<cfl<=1*/
	double* h0; //solucion h en el instante n
	double* h1; //solucion h en el instasnte n+1
    double* q10; //solucion q1 en el instante n
	double* q11; //solucion q1 en el instasnte n+1
    double* q20; //solucion q2 en el instante n
	double* q21; //solucion q2 en el instasnte n+1
	double* x; //discretizacion del dominio espacial
    double* y; //discretizacion del dominio espacial
    double* aux; // puntero auxiliar para el intercambio de datos
    //creacion de los arrays
    h0=new double[(npx+1)*(npy+1)];
    h1=new double[(npx+1)*(npy+1)];
    q10=new double[(npx+1)*(npy+1)];
    q11=new double[(npx+1)*(npy+1)];
    q20=new double[(npx+1)*(npy+1)];
    q21=new double[(npx+1)*(npy+1)];
    x=new double[npx+1];
    y=new double[npy+1];
    
	dx=(b-a)/double(npx);
    dy=(d-c)/double(npy);
    max1 = 0.0;
    max2 = 0.0;   
	
	/* particion x*/
    for (int i=0; i<=npx; i++) {
		x[i]=a+dx*double(i);
    }
    for (int j=0; j<=npy; j++) {
        y[j]=c+dy*double(j);
    }
    for (int i=0; i<=npy; i++) {
        for (int j=0; j<=npx; j++) {
            int ij=i*(npx+1)+j;

            h0[ij]=h_inicial(x[j],y[i]);
            h1[ij]=h0[ij];

            q10[ij]=q1_inicial(x[j],y[i]);
            q11[ij]=q10[ij];

            q20[ij]=q2_inicial(x[j],y[i]);
            q21[ij]=q20[ij];

            double max1_new = fabs(q11[ij]/h1[ij]) + sqrt(9.81 * h1[ij]);
            max1 = max(max1, max1_new);

            double max2_new = fabs(q21[ij]/h1[ij]) + sqrt(9.81 * h1[ij]);
            max2 = max(max2, max2_new);
        }
    }
    
    dt=0.5*cfl*min(dx/max1, dy/max2);
	printf("Dt: %12.8f\n",dt);
	tiempo=0.0;
    tg=dtg;
    
	fp=fopen(fichero_salida,"wt");
	fprintf(fp,"%12.8f %12.8f %5i %12.8f %12.8f %5i",a,b,npx,c,d,npy);
	fprintf(fp,"\n");
    
	fprintf(fp,"%15.8f",tiempo);
	for (int i=0; i<=npy; i++) {
		for (int j=0; j<=npx; j++) {
            int ij=i*(npx+1)+j;
			fprintf(fp,"%15.8f",h0[ij]);
		}
	}
	fprintf(fp,"\n");

    int ij, ip1j, im1j, ijp1, ijm1;
    
	while (tiempo<T) {
        max1 = 0.0;
        max2 = 0.0;
		//bucle en tiempo
        printf("Tiempo: %12.8f\n",tiempo+dt);
        for (int i=1; i<npy; i++) {
            for (int j=1; j<npx; j++) { 
                ij=i*(npx+1)+j;
                ip1j=(i+1)*(npx+1)+j;
                im1j=(i-1)*(npx+1)+j;
                ijp1 = ij+1;
                ijm1 = ij-1;
                
                h1[ij] = h0[ij]
                            - 0.5*dt/dx*(q10[ijp1]-q10[ijm1]-0.5*cfl*dx/dt*(h0[ijm1]-2*h0[ij]+h0[ijp1]))
                            - 0.5*dt/dy*(q20[ip1j]-q20[im1j]-0.5*cfl*dy/dt*(h0[im1j]-2*h0[ij]+h0[ip1j]));

                q11[ij] = q10[ij] 
                            - 0.5*dt/dx*((q10[ijp1]*q10[ijp1]/h0[ijp1] + 0.5*9.81*h0[ijp1]*h0[ijp1]) - (q10[ijm1]*q10[ijm1]/h0[ijm1] + 0.5*9.81*h0[ijm1]*h0[ijm1]) - 0.5*cfl*dx/dt*(q10[ijp1] -2*q10[ij] + q10[ijm1]))
                            - 0.5*dt/dy*((q10[ip1j]*q20[ip1j]/h0[ip1j]) - (q10[im1j]*q20[im1j]/h0[im1j]) - 0.5*cfl*dy/dt*(q10[ip1j] -2*q10[ij] + q10[im1j]));

                q21[ij] = q20[ij] 
                            - 0.5*dt/dx*((q10[ijp1]*q20[ijp1]/h0[ijp1]) - (q10[ijm1]*q20[ijm1]/h0[ijm1]) - 0.5*cfl*dx/dt*(q20[ijp1] -2*q20[ij] + q20[ijm1]))
                            - 0.5*dt/dy*((q20[ip1j]*q20[ip1j]/h0[ip1j] + 0.5*9.81*h0[ip1j]*h0[ip1j]) - (q20[im1j]*q20[im1j]/h0[im1j] + 0.5*9.81*h0[im1j]*h0[im1j]) - 0.5*cfl*dy/dt*(q20[ip1j] -2*q20[ij] + q20[im1j]));

                double max1_new = fabs(q11[ij]/h1[ij]) + sqrt(9.81 * h1[ij]);
                if (max1 < max1_new) max1 = max1_new;

                double max2_new = fabs(q21[ij]/h1[ij]) + sqrt(9.81 * h1[ij]);
                if (max2 < max2_new) max2 = max2_new;
            }
        }
        
        // cond contorno fila 0 y npy
        for (int j=1; j<npx; j++) {
            // caso fila 0:
            ij=j; // =0*(npx+1)+j
            ip1j=(npx+1)+j; // =(0+1)*(npx+1)+j;
            im1j= npy*(npx+1)+j; // Condicion periodica 
            ijp1 = ij+1;
            ijm1 = ij-1;

            h1[ij] = h0[ij]
                         - 0.5*dt/dx*(q10[ijp1]-q10[ijm1]-0.5*cfl*dx/dt*(h0[ijm1]-2*h0[ij]+h0[ijp1]))
                         - 0.5*dt/dy*(q20[ip1j]-q20[im1j]-0.5*cfl*dy/dt*(h0[im1j]-2*h0[ij]+h0[ip1j]));

            q11[ij] = q10[ij] 
                        - 0.5*dt/dx*((q10[ijp1]*q10[ijp1]/h0[ijp1] + 0.5*9.81*h0[ijp1]*h0[ijp1]) - (q10[ijm1]*q10[ijm1]/h0[ijm1] + 0.5*9.81*h0[ijm1]*h0[ijm1]) - 0.5*cfl*dx/dt*(q10[ijp1] -2*q10[ij] + q10[ijm1]))
                        - 0.5*dt/dy*((q10[ip1j]*q20[ip1j]/h0[ip1j]) - (q10[im1j]*q20[im1j]/h0[im1j]) - 0.5*cfl*dy/dt*(q10[ip1j] -2*q10[ij] + q10[im1j]));

            q21[ij] = q20[ij] 
                        - 0.5*dt/dx*((q10[ijp1]*q20[ijp1]/h0[ijp1]) - (q10[ijm1]*q20[ijm1]/h0[ijm1]) - 0.5*cfl*dx/dt*(q20[ijp1] -2*q20[ij] + q20[ijm1]))
                        - 0.5*dt/dy*((q20[ip1j]*q20[ip1j]/h0[ip1j] + 0.5*9.81*h0[ip1j]*h0[ip1j]) - (q20[im1j]*q20[im1j]/h0[im1j] + 0.5*9.81*h0[im1j]*h0[im1j]) - 0.5*cfl*dy/dt*(q20[ip1j] -2*q20[ij] + q20[im1j]));

            double max1_new = fabs(q11[ij]/h1[ij]) + sqrt(9.81 * h1[ij]);
            if (max1 < max1_new) max1 = max1_new;

            double max2_new = fabs(q21[ij]/h1[ij]) + sqrt(9.81 * h1[ij]);
            if (max2 < max2_new) max2 = max2_new;

            // caso fila ny:

            ij=npy*(npx+1)+j;
            ip1j=j; // Condicion periodica
            im1j=(npy-1)*(npx+1)+j;
            ijp1 = ij+1;
            ijm1 = ij-1;
            h1[ij] = h0[ij]
                         - 0.5*dt/dx*(q10[ijp1]-q10[ijm1]-0.5*cfl*dx/dt*(h0[ijm1]-2*h0[ij]+h0[ijp1]))
                         - 0.5*dt/dy*(q20[ip1j]-q20[im1j]-0.5*cfl*dy/dt*(h0[im1j]-2*h0[ij]+h0[ip1j]));

            q11[ij] = q10[ij] 
                        - 0.5*dt/dx*((q10[ijp1]*q10[ijp1]/h0[ijp1] + 0.5*9.81*h0[ijp1]*h0[ijp1]) - (q10[ijm1]*q10[ijm1]/h0[ijm1] + 0.5*9.81*h0[ijm1]*h0[ijm1]) - 0.5*cfl*dx/dt*(q10[ijp1] -2*q10[ij] + q10[ijm1]))
                        - 0.5*dt/dy*((q10[ip1j]*q20[ip1j]/h0[ip1j]) - (q10[im1j]*q20[im1j]/h0[im1j]) - 0.5*cfl*dy/dt*(q10[ip1j] -2*q10[ij] + q10[im1j]));

            q21[ij] = q20[ij] 
                        - 0.5*dt/dx*((q10[ijp1]*q20[ijp1]/h0[ijp1]) - (q10[ijm1]*q20[ijm1]/h0[ijm1]) - 0.5*cfl*dx/dt*(q20[ijp1] -2*q20[ij] + q20[ijm1]))
                        - 0.5*dt/dy*((q20[ip1j]*q20[ip1j]/h0[ip1j] + 0.5*9.81*h0[ip1j]*h0[ip1j]) - (q20[im1j]*q20[im1j]/h0[im1j] + 0.5*9.81*h0[im1j]*h0[im1j]) - 0.5*cfl*dy/dt*(q20[ip1j] -2*q20[ij] + q20[im1j]));

            max1_new = fabs(q11[ij]/h1[ij]) + sqrt(9.81 * h1[ij]);
            if (max1 < max1_new) max1 = max1_new;

            max2_new = fabs(q21[ij]/h1[ij]) + sqrt(9.81 * h1[ij]);
            if (max2 < max2_new) max2 = max2_new;

        }
        // cond contorno columna 0 y npx
        for (int i=1; i<npy;i++) {
            // caso columna 0
            ij=i*(npx+1); // =i*(npx+1)+0;
            ip1j=(i+1)*(npx+1); // = (i+1)*(npx+1)+0;
            im1j=(i-1)*(npx+1); // = (i-1)*(npx+1)+0;
            ijp1 = ij+1;
            ijm1 = i*(npx+1)+npx ; // Condicion periodica
            
            h1[ij] = h0[ij]
                         - 0.5*dt/dx*(q10[ijp1]-q10[ijm1]-0.5*cfl*dx/dt*(h0[ijm1]-2*h0[ij]+h0[ijp1]))
                         - 0.5*dt/dy*(q20[ip1j]-q20[im1j]-0.5*cfl*dy/dt*(h0[im1j]-2*h0[ij]+h0[ip1j]));

            q11[ij] = q10[ij] 
                        - 0.5*dt/dx*((q10[ijp1]*q10[ijp1]/h0[ijp1] + 0.5*9.81*h0[ijp1]*h0[ijp1]) - (q10[ijm1]*q10[ijm1]/h0[ijm1] + 0.5*9.81*h0[ijm1]*h0[ijm1]) - 0.5*cfl*dx/dt*(q10[ijp1] -2*q10[ij] + q10[ijm1]))
                        - 0.5*dt/dy*((q10[ip1j]*q20[ip1j]/h0[ip1j]) - (q10[im1j]*q20[im1j]/h0[im1j]) - 0.5*cfl*dy/dt*(q10[ip1j] -2*q10[ij] + q10[im1j]));

            q21[ij] = q20[ij] 
                        - 0.5*dt/dx*((q10[ijp1]*q20[ijp1]/h0[ijp1]) - (q10[ijm1]*q20[ijm1]/h0[ijm1]) - 0.5*cfl*dx/dt*(q20[ijp1] -2*q20[ij] + q20[ijm1]))
                        - 0.5*dt/dy*((q20[ip1j]*q20[ip1j]/h0[ip1j] + 0.5*9.81*h0[ip1j]*h0[ip1j]) - (q20[im1j]*q20[im1j]/h0[im1j] + 0.5*9.81*h0[im1j]*h0[im1j]) - 0.5*cfl*dy/dt*(q20[ip1j] -2*q20[ij] + q20[im1j]));

            double max1_new = fabs(q11[ij]/h1[ij]) + sqrt(9.81 * h1[ij]);
            if (max1 < max1_new) max1 = max1_new;

            double max2_new = fabs(q21[ij]/h1[ij]) + sqrt(9.81 * h1[ij]);
            if (max2 < max2_new) max2 = max2_new;

            // caso columna npx
            ij=i*(npx+1)+npx; 
            ip1j=(i+1)*(npx+1)+npx; 
            im1j=(i-1)*(npx+1)+npx; 
            ijp1 = i*(npx+1); // Condicion periodica 
            ijm1 = ij-1;

            h1[ij] = h0[ij]
                         - 0.5*dt/dx*(q10[ijp1]-q10[ijm1]-0.5*cfl*dx/dt*(h0[ijm1]-2*h0[ij]+h0[ijp1]))
                         - 0.5*dt/dy*(q20[ip1j]-q20[im1j]-0.5*cfl*dy/dt*(h0[im1j]-2*h0[ij]+h0[ip1j]));

            q11[ij] = q10[ij] 
                        - 0.5*dt/dx*((q10[ijp1]*q10[ijp1]/h0[ijp1] + 0.5*9.81*h0[ijp1]*h0[ijp1]) - (q10[ijm1]*q10[ijm1]/h0[ijm1] + 0.5*9.81*h0[ijm1]*h0[ijm1]) - 0.5*cfl*dx/dt*(q10[ijp1] -2*q10[ij] + q10[ijm1]))
                        - 0.5*dt/dy*((q10[ip1j]*q20[ip1j]/h0[ip1j]) - (q10[im1j]*q20[im1j]/h0[im1j]) - 0.5*cfl*dy/dt*(q10[ip1j] -2*q10[ij] + q10[im1j]));

            q21[ij] = q20[ij] 
                        - 0.5*dt/dx*((q10[ijp1]*q20[ijp1]/h0[ijp1]) - (q10[ijm1]*q20[ijm1]/h0[ijm1]) - 0.5*cfl*dx/dt*(q20[ijp1] -2*q20[ij] + q20[ijm1]))
                        - 0.5*dt/dy*((q20[ip1j]*q20[ip1j]/h0[ip1j] + 0.5*9.81*h0[ip1j]*h0[ip1j]) - (q20[im1j]*q20[im1j]/h0[im1j] + 0.5*9.81*h0[im1j]*h0[im1j]) - 0.5*cfl*dy/dt*(q20[ip1j] -2*q20[ij] + q20[im1j]));

            max1_new = fabs(q11[ij]/h1[ij]) + sqrt(9.81 * h1[ij]);
            if (max1 < max1_new) max1 = max1_new;

            max2_new = fabs(q21[ij]/h1[ij]) + sqrt(9.81 * h1[ij]);
            if (max2 < max2_new) max2 = max2_new;
        }
        // quedan los 4 vertices del rectangulo
        {
            int i=0;
            int j=0;

            ij=i*(npx+1)+j;
            ip1j=(i+1)*(npx+1)+j;
            im1j=npy*(npx+1)+j; // Condicion periodica
            ijp1 = ij+1;
            ijm1 = i*(npx+1)+npx; // Condicion periodica
                
            h1[ij] = h0[ij]
                         - 0.5*dt/dx*(q10[ijp1]-q10[ijm1]-0.5*cfl*dx/dt*(h0[ijm1]-2*h0[ij]+h0[ijp1]))
                         - 0.5*dt/dy*(q20[ip1j]-q20[im1j]-0.5*cfl*dy/dt*(h0[im1j]-2*h0[ij]+h0[ip1j]));

            q11[ij] = q10[ij] 
                        - 0.5*dt/dx*((q10[ijp1]*q10[ijp1]/h0[ijp1] + 0.5*9.81*h0[ijp1]*h0[ijp1]) - (q10[ijm1]*q10[ijm1]/h0[ijm1] + 0.5*9.81*h0[ijm1]*h0[ijm1]) - 0.5*cfl*dx/dt*(q10[ijp1] -2*q10[ij] + q10[ijm1]))
                        - 0.5*dt/dy*((q10[ip1j]*q20[ip1j]/h0[ip1j]) - (q10[im1j]*q20[im1j]/h0[im1j]) - 0.5*cfl*dy/dt*(q10[ip1j] -2*q10[ij] + q10[im1j]));

            q21[ij] = q20[ij] 
                        - 0.5*dt/dx*((q10[ijp1]*q20[ijp1]/h0[ijp1]) - (q10[ijm1]*q20[ijm1]/h0[ijm1]) - 0.5*cfl*dx/dt*(q20[ijp1] -2*q20[ij] + q20[ijm1]))
                        - 0.5*dt/dy*((q20[ip1j]*q20[ip1j]/h0[ip1j] + 0.5*9.81*h0[ip1j]*h0[ip1j]) - (q20[im1j]*q20[im1j]/h0[im1j] + 0.5*9.81*h0[im1j]*h0[im1j]) - 0.5*cfl*dy/dt*(q20[ip1j] -2*q20[ij] + q20[im1j]));

            double max1_new = fabs(q11[ij]/h1[ij]) + sqrt(9.81 * h1[ij]);
            if (max1 < max1_new) max1 = max1_new;

            double max2_new = fabs(q21[ij]/h1[ij]) + sqrt(9.81 * h1[ij]);
            if (max2 < max2_new) max2 = max2_new;

            i=0;
            j=npx;

            ij=i*(npx+1)+j;
            ip1j=(i+1)*(npx+1)+j; 
            im1j=npy*(npx+1)+j; // Condicion periodica 
            ijp1 = i*(npx+1);   // Condicion periodica
            ijm1 = ij-1;
                
            h1[ij] = h0[ij]
                         - 0.5*dt/dx*(q10[ijp1]-q10[ijm1]-0.5*cfl*dx/dt*(h0[ijm1]-2*h0[ij]+h0[ijp1]))
                         - 0.5*dt/dy*(q20[ip1j]-q20[im1j]-0.5*cfl*dy/dt*(h0[im1j]-2*h0[ij]+h0[ip1j]));

            q11[ij] = q10[ij] 
                        - 0.5*dt/dx*((q10[ijp1]*q10[ijp1]/h0[ijp1] + 0.5*9.81*h0[ijp1]*h0[ijp1]) - (q10[ijm1]*q10[ijm1]/h0[ijm1] + 0.5*9.81*h0[ijm1]*h0[ijm1]) - 0.5*cfl*dx/dt*(q10[ijp1] -2*q10[ij] + q10[ijm1]))
                        - 0.5*dt/dy*((q10[ip1j]*q20[ip1j]/h0[ip1j]) - (q10[im1j]*q20[im1j]/h0[im1j]) - 0.5*cfl*dy/dt*(q10[ip1j] -2*q10[ij] + q10[im1j]));

            q21[ij] = q20[ij] 
                        - 0.5*dt/dx*((q10[ijp1]*q20[ijp1]/h0[ijp1]) - (q10[ijm1]*q20[ijm1]/h0[ijm1]) - 0.5*cfl*dx/dt*(q20[ijp1] -2*q20[ij] + q20[ijm1]))
                        - 0.5*dt/dy*((q20[ip1j]*q20[ip1j]/h0[ip1j] + 0.5*9.81*h0[ip1j]*h0[ip1j]) - (q20[im1j]*q20[im1j]/h0[im1j] + 0.5*9.81*h0[im1j]*h0[im1j]) - 0.5*cfl*dy/dt*(q20[ip1j] -2*q20[ij] + q20[im1j]));

            max1_new = fabs(q11[ij]/h1[ij]) + sqrt(9.81 * h1[ij]);
            if (max1 < max1_new) max1 = max1_new;

            max2_new = fabs(q21[ij]/h1[ij]) + sqrt(9.81 * h1[ij]);
            if (max2 < max2_new) max2 = max2_new;

            i=npy;
            j=0;

            ij=i*(npx+1)+j;
            ip1j=j; // Condicion periodica
            im1j=(i-1)*(npx+1)+j; 
            ijp1 = ij+1;
            ijm1 = i*(npx+1)+npx; // Condicion periodica
                
            h1[ij] = h0[ij]
                         - 0.5*dt/dx*(q10[ijp1]-q10[ijm1]-0.5*cfl*dx/dt*(h0[ijm1]-2*h0[ij]+h0[ijp1]))
                         - 0.5*dt/dy*(q20[ip1j]-q20[im1j]-0.5*cfl*dy/dt*(h0[im1j]-2*h0[ij]+h0[ip1j]));

            q11[ij] = q10[ij] 
                        - 0.5*dt/dx*((q10[ijp1]*q10[ijp1]/h0[ijp1] + 0.5*9.81*h0[ijp1]*h0[ijp1]) - (q10[ijm1]*q10[ijm1]/h0[ijm1] + 0.5*9.81*h0[ijm1]*h0[ijm1]) - 0.5*cfl*dx/dt*(q10[ijp1] -2*q10[ij] + q10[ijm1]))
                        - 0.5*dt/dy*((q10[ip1j]*q20[ip1j]/h0[ip1j]) - (q10[im1j]*q20[im1j]/h0[im1j]) - 0.5*cfl*dy/dt*(q10[ip1j] -2*q10[ij] + q10[im1j]));

            q21[ij] = q20[ij] 
                        - 0.5*dt/dx*((q10[ijp1]*q20[ijp1]/h0[ijp1]) - (q10[ijm1]*q20[ijm1]/h0[ijm1]) - 0.5*cfl*dx/dt*(q20[ijp1] -2*q20[ij] + q20[ijm1]))
                        - 0.5*dt/dy*((q20[ip1j]*q20[ip1j]/h0[ip1j] + 0.5*9.81*h0[ip1j]*h0[ip1j]) - (q20[im1j]*q20[im1j]/h0[im1j] + 0.5*9.81*h0[im1j]*h0[im1j]) - 0.5*cfl*dy/dt*(q20[ip1j] -2*q20[ij] + q20[im1j]));

            max1_new = fabs(q11[ij]/h1[ij]) + sqrt(9.81 * h1[ij]);
            if (max1 < max1_new) max1 = max1_new;

            max2_new = fabs(q21[ij]/h1[ij]) + sqrt(9.81 * h1[ij]);
            if (max2 < max2_new) max2 = max2_new;

            i=npy;
            j=npx;

            ij=i*(npx+1)+j;
            ip1j=j; // Condicion periodica
            im1j=(i-1)*(npx+1)+j;
            ijp1 = i*(npx+1); // Condicion periodica
            ijm1 = ij-1;
                
            h1[ij] = h0[ij]
                         - 0.5*dt/dx*(q10[ijp1]-q10[ijm1]-0.5*cfl*dx/dt*(h0[ijm1]-2*h0[ij]+h0[ijp1]))
                         - 0.5*dt/dy*(q20[ip1j]-q20[im1j]-0.5*cfl*dy/dt*(h0[im1j]-2*h0[ij]+h0[ip1j]));

            q11[ij] = q10[ij] 
                        - 0.5*dt/dx*((q10[ijp1]*q10[ijp1]/h0[ijp1] + 0.5*9.81*h0[ijp1]*h0[ijp1]) - (q10[ijm1]*q10[ijm1]/h0[ijm1] + 0.5*9.81*h0[ijm1]*h0[ijm1]) - 0.5*cfl*dx/dt*(q10[ijp1] -2*q10[ij] + q10[ijm1]))
                        - 0.5*dt/dy*((q10[ip1j]*q20[ip1j]/h0[ip1j]) - (q10[im1j]*q20[im1j]/h0[im1j]) - 0.5*cfl*dy/dt*(q10[ip1j] -2*q10[ij] + q10[im1j]));

            q21[ij] = q20[ij] 
                        - 0.5*dt/dx*((q10[ijp1]*q20[ijp1]/h0[ijp1]) - (q10[ijm1]*q20[ijm1]/h0[ijm1]) - 0.5*cfl*dx/dt*(q20[ijp1] -2*q20[ij] + q20[ijm1]))
                        - 0.5*dt/dy*((q20[ip1j]*q20[ip1j]/h0[ip1j] + 0.5*9.81*h0[ip1j]*h0[ip1j]) - (q20[im1j]*q20[im1j]/h0[im1j] + 0.5*9.81*h0[im1j]*h0[im1j]) - 0.5*cfl*dy/dt*(q20[ip1j] -2*q20[ij] + q20[im1j]));

            max1_new = fabs(q11[ij]/h1[ij]) + sqrt(9.81 * h1[ij]);
            if (max1 < max1_new) max1 = max1_new;

            max2_new = fabs(q21[ij]/h1[ij]) + sqrt(9.81 * h1[ij]);
            if (max2 < max2_new) max2 = max2_new;
            
        }

		aux=h0;
        h0=h1;
        h1=aux;
        aux=q10;
        q10=q11;
        q11=aux;
        aux=q20;
        q20=q21;
        q21=aux;
        dt = 0.5*cfl*min(dx/max1,dy/max2);
		tiempo+=dt;
        if (tiempo>=tg) {
            fprintf(fp,"%15.8f",tiempo);
            for (int i=0; i<=npy; i++) {
                for (int j=0; j<=npx; j++) {
                    int ij=i*(npx+1)+j;
                    fprintf(fp,"%15.8f",h0[ij]);
                }
            }
            fprintf(fp,"\n");
            tg+=dtg;
        }
        

	}
    t_fin=clock();
    secs = (double)(t_fin - t_ini) / CLOCKS_PER_SEC;
    printf("%.16g ms\n", secs * 1000.0);
	fclose(fp);
    delete[] h0;
    delete[] h1;
    delete[] q10;
    delete[] q11;
    delete[] q20;
    delete[] q21;
    delete[] x;
    delete[] y;
	
	return 0;
}
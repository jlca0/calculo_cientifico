#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <time.h>
#include <omp.h>

using namespace std;

double hcond_inicial(double x0) {
	/* Aqui podemos definir una funcion de x*/
	double ci;
	ci=1+exp(-x0*x0);
	return ci;
}

double qcond_inicial(double x0) {
	/* Aqui podemos definir una funcion de x*/
	return 0;
}

int main(int argc, char **argv) {
	double a,b,T,nu,cfl,tiempo,tg,dtg;
	int npx,i,j;
    int nh;
	char *hfichero_salida, *qfichero_salida;	
	double dx,dt,max0,g;
	FILE *fpq, *fph;
    clock_t t_ini, t_fin;
    double secs;
    t_ini = clock();

	if (argc != 10) {
		printf("Uso:\n");
		printf(argv[0]);
		printf(" a b T npx cfl nu fichero salida Dt guardado\n");
		printf("a: Comienzo del intervalo.\n");
		printf("b: Final del inervalo.\n");
		printf("T: Tiempo total de integracion.\n");
		printf("npx: N. de particiones del intervalo [a,b]\n");
		printf("cfl: Coef. estabilidad.\n");
		printf("Nombre fichero de salida para h.\n");
        printf("Nombre fichero de salida para q.\n");
        printf("Dt guardado.\n");
        printf("nh: Número de hebras.\n");
		return -1;
	}
	a=atof(argv[1]);
	b=atof(argv[2]);
	T=atof(argv[3]);
	npx=atoi(argv[4]);
	cfl=atof(argv[5]);
	hfichero_salida=argv[6];
    qfichero_salida=argv[7];
    dtg=atof(argv[8]);
    nh=atoi(argv[9]);
    omp_set_num_threads(nh);
	double* hsol0; //solucion de h en el instante n
    double* hsol1;// solucion de h en el instante n+1
    double* qsol0; //solucion de q en el instante n
    double* qsol1;// solucion de q en el instante n+1 
    double* x; //particion del intervalo
    double* aux;
    max0 = 0.0;
    g = 9.81;

    // definicion de los tableros
    hsol0=new double[npx+1];
    hsol1=new double[npx+1];
    qsol0=new double[npx+1];
    qsol1=new double[npx+1];
    x=new double[npx+1];  
    dx=(b-a)/double(npx);

	
    // particion x
    #pragma omp parallel for reduction(max:max0)
    for (i=0; i<=npx; i++) {
		x[i]=a+dx*double(i);
		hsol0[i]=hcond_inicial(x[i]);
        qsol0[i]=qcond_inicial(x[i]);
		if(0 < i < npx){
			max0 = max(fabs(qcond_inicial(x[i])/hcond_inicial(x[i]))+sqrt(g*hcond_inicial(x[i])),fabs(qcond_inicial(x[i-1])/hcond_inicial(x[i-1]))+sqrt(g*hcond_inicial(x[i-1])));
		}
	}

	dt = cfl * dx / max0;
	tiempo=0.0;
    tg=dtg;

	fph=fopen(hfichero_salida,"wt");
    fpq=fopen(qfichero_salida,"wt");
	for (i=0; i<=npx; i++) {
		fprintf(fph,"%12.8f",x[i]);
        fprintf(fpq,"%12.8f",x[i]);
	}
	fprintf(fph,"\n");
    fprintf(fpq,"\n");

	fprintf(fph,"%12.8f",tiempo);
    fprintf(fpq,"%12.8f",tiempo);
	for (i=0; i<=npx; i++) {
		fprintf(fph,"%12.8f",hsol0[i]);
        fprintf(fpq,"%12.8f",qsol0[i]);
	}
	fprintf(fph,"\n");
    fprintf(fpq,"\n");
	
	while (tiempo<T) {
		//bucle en tiempo
		printf("Tiempo: %12.8f\n",tiempo+dt);
        #pragma omp parallel for 
		for (i=1; i<npx; i++) {
			//bucle en espacio
            hsol1[i]=hsol0[i]-0.5*dt/dx*(qsol0[i+1]-qsol0[i-1])+cfl*0.5*(hsol0[i+1]-2*hsol0[i]+hsol0[i-1]);
            qsol1[i]=qsol0[i]-0.5*dt/dx*(qsol0[i+1]*qsol0[i+1]/hsol0[i+1]-qsol0[i-1]*qsol0[i-1]/hsol0[i-1]+g*0.5*(hsol0[i+1]*hsol0[i+1]-hsol0[i-1]*hsol0[i-1]))+cfl*0.5*(qsol0[i+1]-2*qsol0[i]+qsol0[i-1]);        }
        
        // condiciones de contorno
        hsol1[0]=hsol0[0]-0.5*dt/dx*(qsol0[1]-qsol0[npx-1])+cfl*0.5*(hsol0[1]-2*hsol0[0]+hsol0[npx-1]);
		hsol1[npx]=hsol0[npx]-0.5*dt/dx*(qsol0[1]-qsol0[npx-1])+cfl*0.5*(hsol0[1]-2*hsol0[npx]+hsol0[npx-1]);
        aux=hsol0;
        hsol0=hsol1;
        hsol1=aux;

        qsol1[0]=qsol0[0]-0.5*dt/dx*(qsol0[1]*qsol0[1]/hsol0[1]-qsol0[npx-1]*qsol0[npx-1]/hsol0[npx-1]+g*0.5*(hsol0[1]*hsol0[1]-hsol0[npx-1]*hsol0[npx-1]))+cfl*0.5*(qsol0[1]-2*qsol0[0]+qsol0[npx-1]);
        qsol1[npx]=qsol0[npx]-0.5*dt/dx*(qsol0[1]*qsol0[1]/hsol0[1]-qsol0[npx-1]*qsol0[npx-1]/hsol0[npx-1]+g*0.5*(hsol0[1]*hsol0[1]-hsol0[npx-1]*hsol0[npx-1]))+cfl*0.5*(qsol0[1]-2*qsol0[npx]+qsol0[npx-1]);
        aux=qsol0;
        qsol0=qsol1;
        qsol1=aux;

        // adaptación del paso de malla
        max0 = 0.0;
        #pragma omp parallel for reduction(max:max0)
		for (i=0; i<=npx; i++){
    		if (qsol0[i]/hsol0[i] + sqrt(g*hsol0[i]) > max0) max0 = qsol0[i]/hsol0[i] + sqrt(g*hsol0[i]);
		}

		dt=cfl * dx / max0;
		tiempo+=dt;

        // grabamos en el fichero
        if (tiempo>=tg) {
            fprintf(fph,"%12.8f",tiempo);
            fprintf(fpq,"%12.8f",tiempo);
            for (i=0; i<=npx; i++) {
                fprintf(fph,"%12.8f",hsol0[i]);
                fprintf(fpq,"%12.8f",qsol0[i]);
            }
            fprintf(fph,"\n");
            fprintf(fpq,"\n");
            tg+=dtg;
        }
	}

    // comprobamos el tiempo
    t_fin=clock();
    secs = (double)(t_fin - t_ini) / CLOCKS_PER_SEC;
    printf("%.16g milisegundos\n", secs * 1000.0);

    // liberamos memoria
	fclose(fph);
    fclose(fpq);
	delete[] hsol0;
    delete[] hsol1;
    delete[] qsol0;
    delete[] qsol1;
    delete[] x;
	
	return 0;
}


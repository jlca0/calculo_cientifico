# calculo_cientifico

Colección de prácticas de los cursos de Cálculo Científico y Análisis Numérico de EDPs. Incluye implementaciones en Python, C++ con OpenMP y FreeFEM++.

## Cálculo Científico

Prácticas del curso de Cálculo Científico. Implementaciones en Python y C++ orientadas a la resolución numérica de ecuaciones diferenciales ordinarias y problemas de simulación.

- **Diferencias finitas para problemas de contorno**: resolución de problemas elípticos en 1D, con distintas condiciones de contorno, de forma directa o por penalización. Resolución de la ecuación del calor con theta-método, que incluye como casos particulares el esquema implícito, el explícito y Crank-Nicolson.

- **Diferencias finitas para EDPs no lineales**: Resolución de ecuaciones parabólicas no lineales en 2D mediante diferencias finitas. Se comparan tres métodos iterativos: punto fijo con linealización explícita, punto fijo con linealización implícita y método de Newton con jacobiano exacto.

- **Descomposición de dominio**: Implementación del método de Schwarz con solapamiento para la resolución de problemas de contorno mediante diferencias finitas. Se implementan las variantes para 2, 3 y k subdominios, con diferentes condiciones de contorno.

- **Ecuaciones de aguas someras**: Resolución numérica en 1D y 2D de las ecuaciones de aguas someras mediante diferencias finitas. La implementación en C++ incluye paralelización con OpenMP.

## Análisis Numérico de EDPs

Prácticas del curso de Análisis Numérico de EDPs. Implementaciones en Python y FreeFEM++ orientadas a la resolución de ecuaciones en derivadas parciales.

- **Métodos para EDOs**: Resolución de problemas de contorno mediante el método del tiro, combinado con distintos métodos de punto fijo.

- **Método de elementos finitos**: En `mef.py` se implementa el MEF en dimensión 1 para el espacio P1, incluyendo el ensamblaje de las matrices de rigidez y de masa elemento a elemento.

- **FreeFEM++**: Resolución de problemas de contorno mediante el MEF usando FreeFEM++ en R1 y R2 usando espacios de e. f. P1 y P2.

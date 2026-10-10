#include "Busquedas.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <climits>

// ============================================================
// NOMBRE DEL EQUIPO
// ============================================================

std::string obtenerNombreEquipo()
{
    char nombre[MPI_MAX_PROCESSOR_NAME];
    int longitud = 0;

    MPI_Get_processor_name(
        nombre,
        &longitud
    );

    return std::string(
        nombre,
        longitud
    );
}

// ============================================================
// FECHA Y HORA
// ============================================================

std::string obtenerFechaHora()
{
    std::time_t ahora =
        std::time(nullptr);

    std::tm tiempoLocal;

#ifdef _WIN32

    localtime_s(
        &tiempoLocal,
        &ahora
    );

#else

    localtime_r(
        &ahora,
        &tiempoLocal
    );

#endif

    std::ostringstream salida;

    salida
        << std::setfill('0')
        << std::setw(2)
        << tiempoLocal.tm_mday
        << "/"
        << std::setw(2)
        << tiempoLocal.tm_mon + 1
        << "/"
        << tiempoLocal.tm_year + 1900
        << " "
        << std::setw(2)
        << tiempoLocal.tm_hour
        << ":"
        << std::setw(2)
        << tiempoLocal.tm_min
        << ":"
        << std::setw(2)
        << tiempoLocal.tm_sec;

    return salida.str();
}

// ============================================================
// ESCRIBIR SOLO EN LOG
// ============================================================

void escribirLog(
    std::ofstream& log,
    const std::string& mensaje
)
{
#pragma omp critical(LOG_ARCHIVO)
    {
        if (log.is_open())
        {
            log
                << "["
                << obtenerFechaHora()
                << "] "
                << mensaje
                << std::endl;

            log.flush();
        }
    }
}

// ============================================================
// CONSOLA + LOG
// ============================================================

void imprimirYLog(
    std::ofstream& log,
    const std::string& mensaje
)
{
#pragma omp critical(SALIDA_LOG)
    {
        std::cout
            << mensaje
            << std::endl;

        if (log.is_open())
        {
            log
                << "["
                << obtenerFechaHora()
                << "] "
                << mensaje
                << std::endl;

            log.flush();
        }
    }
}

// ============================================================
// GENERAR ARREGLO ALEATORIO
// ============================================================

void llenarArregloAleatorio(
    int* arreglo,
    long long n,
    int maxValor
)
{
#pragma omp parallel
    {
        unsigned int semilla =
            static_cast<unsigned int>(
                std::time(nullptr)
                + omp_get_thread_num() * 10000
                + omp_get_thread_num()
            );

#pragma omp for schedule(static)

        for (
            long long i = 0;
            i < n;
            i++
            )
        {
            semilla =
                1664525 * semilla
                + 1013904223;

            arreglo[i] =
                static_cast<int>(
                    semilla % maxValor
                ) + 1;
        }
    }
}

// ============================================================
// MOSTRAR ARREGLO
// ============================================================

void mostrarArreglo(
    int* arreglo,
    long long n,
    std::ofstream& log
)
{
    std::ostringstream salida;

    salida << "Arreglo: ";

    for (
        long long i = 0;
        i < n;
        i++
        )
    {
        salida
            << "["
            << i
            << "]="
            << arreglo[i]
            << " ";
    }

    // Para arreglos pequenos se muestra en consola
    // y tambien queda guardado en el log.
    imprimirYLog(
        log,
        salida.str()
    );
}

// ============================================================
// MERGE
// ============================================================

void merge(
    int* arreglo,
    int* auxiliar,
    long long izquierda,
    long long medio,
    long long derecha
)
{
    long long i = izquierda;
    long long j = medio + 1;
    long long k = izquierda;

    while (
        i <= medio &&
        j <= derecha
        )
    {
        if (
            arreglo[i]
            <= arreglo[j]
            )
        {
            auxiliar[k++] =
                arreglo[i++];
        }
        else
        {
            auxiliar[k++] =
                arreglo[j++];
        }
    }

    while (i <= medio)
    {
        auxiliar[k++] =
            arreglo[i++];
    }

    while (j <= derecha)
    {
        auxiliar[k++] =
            arreglo[j++];
    }

    for (
        long long x = izquierda;
        x <= derecha;
        x++
        )
    {
        arreglo[x] =
            auxiliar[x];
    }
}

// ============================================================
// MERGE SORT SECUENCIAL
// ============================================================

void mergeSortSecuencial(
    int* arreglo,
    int* auxiliar,
    long long izquierda,
    long long derecha
)
{
    if (
        izquierda >= derecha
        )
    {
        return;
    }

    long long medio =
        izquierda
        + (derecha - izquierda) / 2;

    mergeSortSecuencial(
        arreglo,
        auxiliar,
        izquierda,
        medio
    );

    mergeSortSecuencial(
        arreglo,
        auxiliar,
        medio + 1,
        derecha
    );

    merge(
        arreglo,
        auxiliar,
        izquierda,
        medio,
        derecha
    );
}

// ============================================================
// MERGE SORT PARALELO RECURSIVO
// ============================================================

void mergeSortParaleloRec(
    int* arreglo,
    int* auxiliar,
    long long izquierda,
    long long derecha,
    int profundidad
)
{
    if (
        izquierda >= derecha
        )
    {
        return;
    }

    // Para bloques pequenos evita crear demasiadas tareas.
    if (
        (derecha - izquierda) < 100000 ||
        profundidad >= 5
        )
    {
        mergeSortSecuencial(
            arreglo,
            auxiliar,
            izquierda,
            derecha
        );

        return;
    }

    long long medio =
        izquierda
        + (derecha - izquierda) / 2;

#pragma omp task shared(arreglo, auxiliar)
    {
        mergeSortParaleloRec(
            arreglo,
            auxiliar,
            izquierda,
            medio,
            profundidad + 1
        );
    }

#pragma omp task shared(arreglo, auxiliar)
    {
        mergeSortParaleloRec(
            arreglo,
            auxiliar,
            medio + 1,
            derecha,
            profundidad + 1
        );
    }

#pragma omp taskwait

    merge(
        arreglo,
        auxiliar,
        izquierda,
        medio,
        derecha
    );
}

// ============================================================
// MERGE SORT PARALELO
// ============================================================

void mergeSortParalelo(
    int* arreglo,
    long long n
)
{
    if (n <= 1)
    {
        return;
    }

    int* auxiliar =
        new int[n];

#pragma omp parallel
    {
#pragma omp single

        {
            mergeSortParaleloRec(
                arreglo,
                auxiliar,
                0,
                n - 1,
                0
            );
        }
    }

    delete[] auxiliar;
}

// ============================================================
// BUSQUEDA SECUENCIAL OPENMP
// ============================================================

long long busquedaLinealOpenMP(
    int* arreglo,
    long long n,
    int target,
    long long offsetGlobal,
    bool detallado,
    std::ofstream& log,
    int rank,
    const std::string& equipo
)
{
    long long posicionEncontrada =
        LLONG_MAX;

    // Guardar inicio en LOG solamente.
    {
        std::ostringstream salida;

        salida
            << "[Equipo: "
            << equipo
            << "] "
            << "[Nodo MPI: "
            << rank
            << "] "
            << "[Busqueda Secuencial OpenMP] "
            << "[Elementos: "
            << n
            << "] "
            << "[Offset Global: "
            << offsetGlobal
            << "] "
            << "[Target: "
            << target
            << "]";

        escribirLog(
            log,
            salida.str()
        );
    }

#pragma omp parallel

    {
        int hilo =
            omp_get_thread_num();

        int totalHilos =
            omp_get_num_threads();

        long long inicio =
            (n * hilo)
            / totalHilos;

        long long fin =
            (n * (hilo + 1))
            / totalHilos
            - 1;

        if (
            inicio < n &&
            fin >= inicio
            )
        {
            long long encontradoHilo =
                -1;

            // =============================================
            // REGISTRAR BLOQUE DEL HILO EN LOG
            // =============================================

            {
                std::ostringstream salida;

                salida
                    << "[Equipo: "
                    << equipo
                    << "] "
                    << "[Nodo MPI: "
                    << rank
                    << "] "
                    << "[Hilo OpenMP: "
                    << hilo
                    << "] "
                    << "[Busqueda: SECUENCIAL] "
                    << "[Bloque local: "
                    << inicio
                    << " - "
                    << fin
                    << "] "
                    << "[Bloque global: "
                    << inicio + offsetGlobal
                    << " - "
                    << fin + offsetGlobal
                    << "] "
                    << "[Target: "
                    << target
                    << "]";

                escribirLog(
                    log,
                    salida.str()
                );
            }

            // =============================================
            // BUSQUEDA
            // =============================================

            for (
                long long i = inicio;
                i <= fin;
                i++
                )
            {
                if (
                    arreglo[i]
                    == target
                    )
                {
                    encontradoHilo =
                        i + offsetGlobal;

                    break;
                }
            }

            // =============================================
            // ENCONTRADO
            // =============================================

            if (
                encontradoHilo != -1
                )
            {
#pragma omp critical(RESULTADO_LINEAL)

                {
                    if (
                        encontradoHilo
                        < posicionEncontrada
                        )
                    {
                        posicionEncontrada =
                            encontradoHilo;
                    }
                }

                std::ostringstream salida;

                salida
                    << "[Equipo: "
                    << equipo
                    << "] "
                    << "[Nodo MPI: "
                    << rank
                    << "] "
                    << "[Hilo OpenMP: "
                    << hilo
                    << "] "
                    << "[Busqueda: SECUENCIAL] "
                    << "[Estado: ENCONTRADO] "
                    << "[Indice Global: "
                    << encontradoHilo
                    << "]";

                escribirLog(
                    log,
                    salida.str()
                );
            }

            // =============================================
            // NO ENCONTRADO
            // =============================================

            else
            {
                std::ostringstream salida;

                salida
                    << "[Equipo: "
                    << equipo
                    << "] "
                    << "[Nodo MPI: "
                    << rank
                    << "] "
                    << "[Hilo OpenMP: "
                    << hilo
                    << "] "
                    << "[Busqueda: SECUENCIAL] "
                    << "[Estado: NO ENCONTRADO] "
                    << "[Bloque global: "
                    << inicio + offsetGlobal
                    << " - "
                    << fin + offsetGlobal
                    << "]";

                escribirLog(
                    log,
                    salida.str()
                );
            }
        }
    }

    if (
        posicionEncontrada
        == LLONG_MAX
        )
    {
        escribirLog(
            log,
            "[Busqueda Secuencial] Resultado final: NO ENCONTRADO"
        );

        return -1;
    }

    {
        std::ostringstream salida;

        salida
            << "[Busqueda Secuencial] "
            << "Resultado final: ENCONTRADO "
            << "[Indice Global: "
            << posicionEncontrada
            << "]";

        escribirLog(
            log,
            salida.str()
        );
    }

    return posicionEncontrada;
}

// ============================================================
// BUSQUEDA BINARIA SIMPLE
// ============================================================

long long busquedaBinariaSimple(
    int* arreglo,
    long long izquierda,
    long long derecha,
    int target
)
{
    long long resultado =
        -1;

    while (
        izquierda <= derecha
        )
    {
        long long medio =
            izquierda
            + (derecha - izquierda) / 2;

        if (
            arreglo[medio]
            == target
            )
        {
            resultado =
                medio;

            // Buscar la primera aparicion.
            derecha =
                medio - 1;
        }

        else if (
            arreglo[medio]
            < target
            )
        {
            izquierda =
                medio + 1;
        }

        else
        {
            derecha =
                medio - 1;
        }
    }

    return resultado;
}

// ============================================================
// BUSQUEDA BINARIA OPENMP
// ============================================================

long long busquedaBinariaOpenMP(
    int* arreglo,
    long long n,
    int target,
    long long offsetGlobal,
    bool detallado,
    std::ofstream& log,
    int rank,
    const std::string& equipo
)
{
    if (n <= 0)
    {
        return -1;
    }

    long long mejorResultado =
        LLONG_MAX;

    {
        std::ostringstream salida;

        salida
            << "[Equipo: "
            << equipo
            << "] "
            << "[Nodo MPI: "
            << rank
            << "] "
            << "[Busqueda Binaria OpenMP] "
            << "[Elementos: "
            << n
            << "] "
            << "[Offset Global: "
            << offsetGlobal
            << "] "
            << "[Target: "
            << target
            << "]";

        escribirLog(
            log,
            salida.str()
        );
    }

#pragma omp parallel

    {
        int hilo =
            omp_get_thread_num();

        int totalHilos =
            omp_get_num_threads();

        long long inicio =
            (n * hilo)
            / totalHilos;

        long long fin =
            (n * (hilo + 1))
            / totalHilos
            - 1;

        if (
            inicio < n &&
            fin >= inicio
            )
        {
            // =============================================
            // REGISTRAR BLOQUE
            // =============================================

            {
                std::ostringstream salida;

                salida
                    << "[Equipo: "
                    << equipo
                    << "] "
                    << "[Nodo MPI: "
                    << rank
                    << "] "
                    << "[Hilo OpenMP: "
                    << hilo
                    << "] "
                    << "[Busqueda: BINARIA] "
                    << "[Bloque local: "
                    << inicio
                    << " - "
                    << fin
                    << "] "
                    << "[Bloque global: "
                    << inicio + offsetGlobal
                    << " - "
                    << fin + offsetGlobal
                    << "] "
                    << "[Valores: "
                    << arreglo[inicio]
                    << " - "
                    << arreglo[fin]
                    << "] "
                    << "[Target: "
                    << target
                    << "]";

                escribirLog(
                    log,
                    salida.str()
                );
            }

            // =============================================
            // TARGET DENTRO DEL RANGO
            // =============================================

            if (
                target >= arreglo[inicio] &&
                target <= arreglo[fin]
                )
            {
                long long resultadoLocal =
                    busquedaBinariaSimple(
                        arreglo,
                        inicio,
                        fin,
                        target
                    );

                if (
                    resultadoLocal != -1
                    )
                {
                    long long indiceGlobal =
                        resultadoLocal
                        + offsetGlobal;

#pragma omp critical(RESULTADO_BINARIO)

                    {
                        if (
                            indiceGlobal
                            < mejorResultado
                            )
                        {
                            mejorResultado =
                                indiceGlobal;
                        }
                    }

                    std::ostringstream salida;

                    salida
                        << "[Equipo: "
                        << equipo
                        << "] "
                        << "[Nodo MPI: "
                        << rank
                        << "] "
                        << "[Hilo OpenMP: "
                        << hilo
                        << "] "
                        << "[Busqueda: BINARIA] "
                        << "[Estado: ENCONTRADO] "
                        << "[Indice Global: "
                        << indiceGlobal
                        << "]";

                    escribirLog(
                        log,
                        salida.str()
                    );
                }
                else
                {
                    std::ostringstream salida;

                    salida
                        << "[Equipo: "
                        << equipo
                        << "] "
                        << "[Nodo MPI: "
                        << rank
                        << "] "
                        << "[Hilo OpenMP: "
                        << hilo
                        << "] "
                        << "[Busqueda: BINARIA] "
                        << "[Estado: NO ENCONTRADO]";

                    escribirLog(
                        log,
                        salida.str()
                    );
                }
            }

            // =============================================
            // TARGET FUERA DEL RANGO
            // =============================================

            else
            {
                std::ostringstream salida;

                salida
                    << "[Equipo: "
                    << equipo
                    << "] "
                    << "[Nodo MPI: "
                    << rank
                    << "] "
                    << "[Hilo OpenMP: "
                    << hilo
                    << "] "
                    << "[Busqueda: BINARIA] "
                    << "[Estado: TARGET FUERA DEL RANGO]";

                escribirLog(
                    log,
                    salida.str()
                );
            }
        }
    }

    if (
        mejorResultado
        == LLONG_MAX
        )
    {
        escribirLog(
            log,
            "[Busqueda Binaria] Resultado final: NO ENCONTRADO"
        );

        return -1;
    }

    {
        std::ostringstream salida;

        salida
            << "[Busqueda Binaria] "
            << "Resultado final: ENCONTRADO "
            << "[Indice Global: "
            << mejorResultado
            << "]";

        escribirLog(
            log,
            salida.str()
        );
    }

    return mejorResultado;
}
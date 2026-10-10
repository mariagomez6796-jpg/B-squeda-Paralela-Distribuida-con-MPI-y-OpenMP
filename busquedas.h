#pragma once

#include <mpi.h>
#include <omp.h>

#include <fstream>
#include <string>

// ============================================================
// INFORMACION / LOGS
// ============================================================

std::string obtenerNombreEquipo();
std::string obtenerFechaHora();

void escribirLog(
    std::ofstream& log,
    const std::string& mensaje
);

void imprimirYLog(
    std::ofstream& log,
    const std::string& mensaje
);

// ============================================================
// ARREGLO
// ============================================================

void llenarArregloAleatorio(
    int* arreglo,
    long long n,
    int maxValor
);

void mostrarArreglo(
    int* arreglo,
    long long n,
    std::ofstream& log
);

// ============================================================
// MERGE SORT
// ============================================================

void merge(
    int* arreglo,
    int* auxiliar,
    long long izquierda,
    long long medio,
    long long derecha
);

void mergeSortSecuencial(
    int* arreglo,
    int* auxiliar,
    long long izquierda,
    long long derecha
);

void mergeSortParaleloRec(
    int* arreglo,
    int* auxiliar,
    long long izquierda,
    long long derecha,
    int profundidad
);

void mergeSortParalelo(
    int* arreglo,
    long long n
);

// ============================================================
// BUSQUEDA SECUENCIAL
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
);

// ============================================================
// BUSQUEDA BINARIA
// ============================================================

long long busquedaBinariaSimple(
    int* arreglo,
    long long izquierda,
    long long derecha,
    int target
);

long long busquedaBinariaOpenMP(
    int* arreglo,
    long long n,
    int target,
    long long offsetGlobal,
    bool detallado,
    std::ofstream& log,
    int rank,
    const std::string& equipo
);
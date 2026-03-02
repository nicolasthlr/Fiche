// ================================================================
//         FICHE RÉVISION BLAS — Fonctions essentielles
// ================================================================

#include <iostream>
#include <cstdlib>
#include <cmath>
#include "cblas.h"
using namespace std;

// Compiler avec : g++ main.cpp -o main -llapack -lblas
// Avec MPI     : mpicxx main.cpp -o main -llapack -lblas

// Convention : remplacer "d" par "z" pour les complexes
// d = double, s = float, z = complex double

// Génération aléatoire (rappel) :
// srand(time(0) + rank);               // initialiser UNE SEULE FOIS (+ rank si MPI)
// double x = (double)rand() / RAND_MAX; // double entre 0 et 1
// double x = ((double)rand() / RAND_MAX) - 0.5; // double entre -0.5 et 0.5


// ================================================================
// PARAMÈTRES COMMUNS
// ================================================================

// n      = nombre d'éléments du vecteur (ou taille du problème)
// inc    = incrément entre éléments (presque toujours 1)
// a, b   = scalaires (double)
// x, y   = double* vecteurs
// A, B,C = double* matrices (stockées en 1D)
// lda    = leading dimension de A
//          → CblasRowMajor : lda = nb de COLONNES
//          → CblasColMajor : lda = nb de LIGNES

// CblasRowMajor = stockage C++ naturel (ligne par ligne)  ← utiliser par défaut
// CblasColMajor = stockage Fortran (colonne par colonne)  ← utilisé par LAPACK
// CblasNoTrans  = pas de transposition
// CblasTrans    = transposer la matrice


// ================================================================
// 1. OPÉRATIONS SUR VECTEURS
// ================================================================

// y <-- a*x + y   (axpy = "a times x plus y")
// Utile pour : mise à jour d'un vecteur, gradient conjugué (x = x + alpha*p)
cblas_daxpy(n, a, x, incx, y, incy);
//   n    = nombre d'éléments
//   a    = scalaire multiplicateur de x
//   x    = vecteur source (non modifié)
//   incx = incrément de x (= 1)
//   y    = vecteur destination → MODIFIÉ (y = a*x + y)
//   incy = incrément de y (= 1)
//
// Exemple : double a = 2.0;
//           cblas_daxpy(N, a, x, 1, y, 1);  // y = 2*x + y


// ----------------------------------------------------------------

// y <-- x   (copie de vecteur)
cblas_dcopy(n, x, incx, y, incy);
//   x = vecteur source
//   y = vecteur destination → MODIFIÉ (copie de x)
//
// Exemple : cblas_dcopy(N, x, 1, y, 1);  // y = x


// ----------------------------------------------------------------

// résultat = x · y   (produit scalaire)
// Retourne un double directement
double dot = cblas_ddot(n, x, incx, y, incy);
//
// Exemple : double dot = cblas_ddot(N, x, 1, y, 1);
//
// Avec MPI : chaque rank calcule son dot local, puis MPI_Reduce(MPI_SUM)


// ----------------------------------------------------------------

// résultat = ||x||₂   (norme euclidienne)
// Retourne un double directement
double nrm = cblas_dnrm2(n, x, incx);
//
// Exemple : double nrm = cblas_dnrm2(N, x, 1);
//
// ⚠️  Avec MPI : ne PAS réduire dnrm2 directement !
//     → Calculer ||x_local||² = ddot(x,x), réduire avec MPI_SUM, puis sqrt()
//     double loc_nrm_sq = cblas_ddot(loc_n, x, 1, x, 1);
//     double nrm_sq;
//     MPI_Reduce(&loc_nrm_sq, &nrm_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//     if (rank == 0) printf("norme = %f\n", sqrt(nrm_sq));


// ----------------------------------------------------------------

// x <-- a*x   (mise à l'échelle d'un vecteur)
cblas_dscal(n, a, x, incx);
//   x = vecteur → MODIFIÉ (x = a*x)
//
// Exemple : cblas_dscal(N, 2.0, x, 1);  // x = 2*x


// ================================================================
// 2. OPÉRATIONS MATRICE-VECTEUR
// ================================================================

// y <-- a*A*x + b*y   (gemv = "general matrix-vector multiply")
cblas_dgemv(Order, TransA, M, N, a, A, lda, x, incx, b, y, incy);
//   Order  = CblasRowMajor (stockage C++ ligne par ligne) ← toujours utiliser ça
//   TransA = CblasNoTrans (pas de transposition) ou CblasTrans
//   M      = nombre de LIGNES de A
//   N      = nombre de COLONNES de A
//   a      = scalaire devant A*x (souvent 1.0)
//   A      = matrice M×N
//   lda    = nb de colonnes de A (en RowMajor) = N
//   x      = vecteur de taille N
//   incx   = 1
//   b      = scalaire devant y (0.0 si on veut juste y = A*x)
//   y      = vecteur résultat de taille M → MODIFIÉ
//   incy   = 1
//
// Exemple (y = A*x, sans terme b*y) :
//   double* y = new double[M]();
//   cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0, A, N, x, 1, 0.0, y, 1);
//
// ⚠️  b=0.0 → y = a*A*x   (le y initial est ignoré)
//     b=1.0 → y = a*A*x + y  (accumulation)


// ================================================================
// 3. OPÉRATIONS MATRICE-MATRICE
// ================================================================

// C <-- a*A*B + b*C   (gemm = "general matrix-matrix multiply")
cblas_dgemm(Order, TransA, TransB, M, N, K, a, A, lda, B, ldb, b, C, ldc);
//   Order  = CblasRowMajor
//   TransA = CblasNoTrans ou CblasTrans
//   TransB = CblasNoTrans ou CblasTrans
//   M      = nb de lignes de A (et de C)
//   N      = nb de colonnes de B (et de C)
//   K      = nb de colonnes de A = nb de lignes de B
//   a      = scalaire devant A*B (souvent 1.0)
//   A      = matrice M×K,  lda = K (en RowMajor)
//   B      = matrice K×N,  ldb = N (en RowMajor)
//   b      = scalaire devant C (0.0 si on veut juste C = A*B)
//   C      = matrice M×N → MODIFIÉE,  ldc = N (en RowMajor)
//
// Exemple (C = A*B) :
//   double* C = new double[M*N]();
//   cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
//               1.0, A, K, B, N, 0.0, C, N);


// ================================================================
// 4. RÉSUMÉ — Quoi utiliser selon la situation
// ================================================================

//  Besoin                          Fonction         Retour
//  ------                          --------         ------
//  Dot product x·y                 cblas_ddot       double
//  Norme ||x||₂                    cblas_dnrm2      double
//  Mise à jour  y = ax + y         cblas_daxpy      void (modifie y)
//  Copie        y = x              cblas_dcopy      void (modifie y)
//  Scaling      x = ax             cblas_dscal      void (modifie x)
//  Matvec       y = aAx + by       cblas_dgemv      void (modifie y)
//  Matmat       C = aAB + bC       cblas_dgemm      void (modifie C)


// ================================================================
// 5. PATTERN TYPE EXAM — BLAS + MPI
// ================================================================

// Dot product distribué :
//   double loc_dot = cblas_ddot(loc_n, x, 1, y, 1);
//   double dot;
//   MPI_Reduce(&loc_dot, &dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

// Norme distribuée :
//   double loc_nrm_sq = cblas_ddot(loc_n, x, 1, x, 1);  // ||x_local||²
//   double nrm_sq;
//   MPI_Reduce(&loc_nrm_sq, &nrm_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//   if (rank == 0) printf("norme = %f\n", sqrt(nrm_sq));

// Matvec distribué (chaque rank a ses lignes de A) :
//   double* y_local = new double[loc_rows]();
//   cblas_dgemv(CblasRowMajor, CblasNoTrans, loc_rows, N, 1.0, A_local, N, x, 1, 0.0, y_local, 1);
//   MPI_Gather(y_local, loc_rows, MPI_DOUBLE, y_global, loc_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);



// ================================================================
//         FICHE RÉVISION LAPACK — Fonctions essentielles
// ================================================================

#include <iostream>
#include <cstdlib>
#include <cmath>
using namespace std;

// ================================================================
// CONVENTION OBLIGATOIRE — Toujours mettre en haut du fichier
// ================================================================

// Macro qui ajoute automatiquement le _ à la fin du nom de fonction
// (convention Fortran, obligatoire pour appeler LAPACK depuis C++)
#define F77NAME(x) x##_

// Toutes les routines LAPACK doivent être déclarées en extern "C"
// pour éviter le name mangling du C++

// Compiler avec : g++ main.cpp -o main -llapack -lblas
// Avec MPI     : mpicxx main.cpp -o main -llapack -lblas


// ================================================================
// RÈGLES GÉNÉRALES LAPACK
// ================================================================

// 1. Tous les arguments sont passés PAR RÉFÉRENCE (const int& n, pas int n)
// 2. Les matrices sont stockées en 1D, COLONNE PAR COLONNE (column-major)
//    → A[i + j*N]   (i=ligne, j=colonne)
// 3. Les matrices sont souvent ÉCRASÉES après l'appel
//    → faire une copie si on en a besoin après
// 4. Toujours vérifier info après l'appel :
//    info = 0  → succès
//    info < 0  → argument invalide
//    info > 0  → échec numérique (ex: matrice singulière)

// Nommage des routines :
// d  = double   (s = float)
// ge = matrice générale   (sy = symétrique)
// sv = solve    (ev = eigenvalues, trf = factorisation LU)


// ================================================================
// DÉCLARATIONS DE VARIABLES — Style du cours
// ================================================================

//    const int N = 3;              // taille de la matrice
//    double* A    = new double[N*N]();  // matrice N×N (le () initialise à 0)
//    double* b    = new double[N]();    // vecteur second membre
//    double* w    = new double[N]();    // valeurs propres
//    int*    ipiv = new int[N]();       // tableau de pivots (TOUJOURS int*, jamais double*)
//    int     info;                      // code de retour
//    int     lwork;                     // taille du workspace


// ================================================================
// REMPLISSAGE D'UNE MATRICE — Column-major (colonne par colonne)
// ================================================================

//  Matrice 3×3 :   | 2  1  1 |
//                  | 4  3  3 |
//                  | 8  7  9 |
//
//    double* A = new double[3*3]();
//    // Colonne 0
//    A[0] = 2.0;  A[1] = 4.0;  A[2] = 8.0;
//    // Colonne 1
//    A[3] = 1.0;  A[4] = 3.0;  A[5] = 7.0;
//    // Colonne 2
//    A[6] = 1.0;  A[7] = 3.0;  A[8] = 9.0;
//
// Formule générale : A[i + j*N]  (ligne i, colonne j)
//
// ⚠️  Pour une matrice SYMÉTRIQUE : A[i + j*N] = A[j + i*N]
//     → remplir les deux triangles identiquement


// ================================================================
// 1. SYSTÈME LINÉAIRE Ax = b  →  dgesv_
// ================================================================

extern "C" {
    void F77NAME(dgesv)(const int& n,     // taille de la matrice A (N×N)
                        const int& nrhs,  // nombre de seconds membres (= 1 en général)
                        double* A,        // matrice N×N → ÉCRASÉE par la facto LU
                        const int& lda,   // leading dimension = N
                        int* ipiv,        // int* ipiv = new int[N]()  ← TOUJOURS int*
                        double* b,        // second membre → ÉCRASÉ par la solution x
                        const int& ldb,   // leading dimension de b = N
                        int& info);       // 0=succès, <0=arg invalide, >0=singulier
}

// EXEMPLE COMPLET :
//
//    const int N    = 3;
//    const int nrhs = 1;
//    double* A    = new double[N*N]();
//    double* b    = new double[N]();
//    int*    ipiv = new int[N]();       // ← int*, pas double* !
//    int     info;
//
//    // Remplir A (column-major) et b ...
//    b[0] = 4.0; b[1] = 10.0; b[2] = 26.0;
//    A[0] = 2.0; A[1] = 4.0; A[2] = 8.0;   // colonne 0
//    A[3] = 1.0; A[4] = 3.0; A[5] = 7.0;   // colonne 1
//    A[6] = 1.0; A[7] = 3.0; A[8] = 9.0;   // colonne 2
//
//    F77NAME(dgesv)(N, nrhs, A, N, ipiv, b, N, info);
//
//    if (info != 0) printf("Erreur dgesv : info = %d\n", info);
//    // b[0], b[1], b[2] contiennent maintenant la solution x
//
//    delete[] A; delete[] b; delete[] ipiv;


// ================================================================
// 2. VALEURS PROPRES d'une matrice SYMÉTRIQUE  →  dsyev_
// ================================================================

extern "C" {
    void F77NAME(dsyev)(const char& jobz,  // 'N' = valeurs propres seulement
                                           // 'V' = valeurs propres + vecteurs propres
                        const char& uplo,  // 'U' = triangle supérieur
                                           // 'L' = triangle inférieur
                        const int& n,      // taille de la matrice
                        double* a,         // matrice N×N → ÉCRASÉE après l'appel
                        const int& lda,    // leading dimension = N
                        double* w,         // double* w = new double[N]() → valeurs propres (ordre croissant)
                        double* work,      // workspace temporaire
                        const int& lwork,  // -1 pour query, puis taille optimale
                        int* info);        // int info (pas int& ici !)
}

// ⚠️  TOUJOURS APPELER EN DEUX FOIS :

// EXEMPLE COMPLET :
//
//    const int N  = 4;
//    char jobz    = 'N';    // valeurs propres seulement
//    char uplo    = 'U';    // triangle supérieur
//    double* A    = new double[N*N]();
//    double* w    = new double[N]();    // contiendra les valeurs propres
//    int     info;
//
//    // Remplir A (column-major, symétrique) ...
//
//    // ÉTAPE 1 : query → demander la taille optimale du workspace
//    int lwork = -1;
//    double work_query;
//    F77NAME(dsyev)(jobz, uplo, N, A, N, w, &work_query, lwork, &info);
//
//    // ÉTAPE 2 : allouer le workspace avec la taille optimale
//    lwork = (int)work_query;
//    double* work = new double[lwork]();
//
//    // ÉTAPE 3 : vrai calcul des valeurs propres
//    F77NAME(dsyev)(jobz, uplo, N, A, N, w, work, lwork, &info);
//
//    if (info != 0) printf("Erreur dsyev : info = %d\n", info);
//    // w[0] <= w[1] <= ... <= w[N-1]  (ordre croissant automatique)
//
//    delete[] A; delete[] w; delete[] work;

// CALCULER LE DÉTERMINANT depuis les valeurs propres :
//    double det = 1.0;
//    for (int k = 0; k < N; k++) det *= w[k];
//    // det(A) = produit de toutes les valeurs propres


// ================================================================
// 3. FACTORISATION LU  →  dgetrf_
// ================================================================

extern "C" {
    void F77NAME(dgetrf)(const int& m,   // nombre de lignes de A
                         const int& n,   // nombre de colonnes de A
                         double* a,      // matrice → ÉCRASÉE par L et U
                         const int& lda, // leading dimension = m
                         int* ipiv,      // int* ipiv = new int[N]()
                         int& info);     // 0=succès, >0=matrice singulière
}

// EXEMPLE COMPLET :
//
//    const int N  = 3;
//    double* A    = new double[N*N]();
//    int*    ipiv = new int[N]();
//    int     info;
//
//    // Remplir A ...
//
//    F77NAME(dgetrf)(N, N, A, N, ipiv, info);
//
//    if (info != 0) printf("Erreur dgetrf : info = %d\n", info);
//    // A contient maintenant L et U combinées
//
//    delete[] A; delete[] ipiv;

// ⚠️  dgetrf_ factorise seulement — pour résoudre directement utiliser dgesv_


// ================================================================
// 4. ERREURS FRÉQUENTES À ÉVITER
// ================================================================

// ❌  int* ipiv = new double[N]();   → ipiv doit être int*, pas double* !
// ✅  int* ipiv = new int[N]();

// ❌  F77NAME(dsyev)(..., lwork, ...);    → appel unique sans query
// ✅  Toujours deux appels : d'abord lwork=-1, puis lwork=taille_optimale

// ❌  A[i*N + j]   → ordre C++ (row-major)
// ✅  A[i + j*N]   → ordre LAPACK (column-major)

// ❌  delete[] A avant d'utiliser les résultats
// ✅  delete[] à la fin seulement


// ================================================================
// 5. PATTERN TYPE EXAM — LAPACK + MPI
// ================================================================

// Chaque rank travaille sur son bloc localement avec LAPACK,
// puis MPI combine les résultats sur rank 0.

//    const int N = 8;
//
//    // 1. Remplir le bloc local
//    double* A = new double[N*N]();
//    srand(time(0) + rank);    // graine différente par rank !
//    for (int i = 0; i < N; i++) {
//        for (int j = i; j < N; j++) {
//            double val = ((double)rand() / RAND_MAX) - 0.5;
//            A[i + j*N] = val;
//            A[j + i*N] = val;   // symétrie
//        }
//    }
//
//    // 2. Valeurs propres avec dsyev_
//    char jobz = 'N', uplo = 'U';
//    double* w = new double[N]();
//    int info, lwork = -1;
//    double work_query;
//    F77NAME(dsyev)(jobz, uplo, N, A, N, w, &work_query, lwork, &info);
//    lwork = (int)work_query;
//    double* work = new double[lwork]();
//    F77NAME(dsyev)(jobz, uplo, N, A, N, w, work, lwork, &info);
//
//    // 3. Déterminant local
//    double loc_det = 1.0;
//    for (int k = 0; k < N; k++) loc_det *= w[k];
//
//    // 4. MPI_Reduce → produit des déterminants locaux sur rank 0
//    double det_global;
//    MPI_Reduce(&loc_det, &det_global, 1, MPI_DOUBLE, MPI_PROD, 0, MPI_COMM_WORLD);
//    //                                                ^^^^^^^^ produit (pas somme !)
//    if (rank == 0) printf("det(A) = %e\n", det_global);
//
//    delete[] A; delete[] w; delete[] work;



// ================================================================
//   FICHE RÉVISION MPI — Commandes clés par exercice
// ================================================================

#include <iostream>
#include <mpi.h>
#include <cmath>
#include <cstdlib>
using namespace std;

// ╔══════════════════════════════════════════════════════════════╗
// ║          SQUELETTE DE BASE (tous les exercices)             ║
// ╚══════════════════════════════════════════════════════════════╝

// Toujours commencer par ça :

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);          // Initialise MPI — TOUJOURS EN PREMIER

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Mon identifiant (commence à 0)
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Nombre total de processus

    // ... code ...

    MPI_Finalize();   // Termine MPI — TOUJOURS EN DERNIER
    return 0;
}

// Compiler :  mpicxx mon_fichier.cpp -o mon_prog
// Exécuter :  mpiexec -np 4 ./mon_prog


// ╔══════════════════════════════════════════════════════════════╗
// ║  EXERCICE 12.1 — Lire une valeur et la distribuer à tous    ║
// ╚══════════════════════════════════════════════════════════════╝

// COMMANDE CLÉ : MPI_Bcast
// → Rank 0 lit la valeur, puis la broadcast à tous les autres processus.
// → Tous les processus appellent MPI_Bcast (même ceux qui reçoivent).

    int value;
    if (rank == 0) {
        cin >> value;       // Seul rank 0 lit l'entrée utilisateur
    }

    // Broadcast : envoie 'value' depuis rank 0 vers TOUS les processus
    MPI_Bcast(
        &value,             // pointeur vers la donnée à envoyer/recevoir
        1,                  // nombre d'éléments
        MPI_INT,            // type MPI
        0,                  // rang du processus source (root)
        MPI_COMM_WORLD      // communicateur
    );

    cout << "Rank " << rank << " : valeur = " << value << endl;

// ⚠️  PATTERN IMPORTANT : seul rank 0 fait cin>>, mais TOUS appellent MPI_Bcast


// ╔══════════════════════════════════════════════════════════════╗
// ║  EXERCICE 12.2 — Passage en anneau (ring)                   ║
// ╚══════════════════════════════════════════════════════════════╝

// COMMANDES CLÉS : MPI_Send + MPI_Recv
// → La valeur circule de rang en rang : 0 → 1 → 2 → ... → (size-1) → 0
// → ATTENTION au deadlock : ne jamais mettre deux Recv face-à-face,
//   ni deux Send face-à-face sans Recv correspondant.

    int left  = (rank - 1 + size) % size;  // voisin gauche (avec wrap-around)
    int right = (rank + 1) % size;          // voisin droit  (avec wrap-around)

    // Envoi bloquant : attend que le buffer soit copié avant de continuer
    MPI_Send(
        &value,             // pointeur vers la donnée à envoyer
        1,                  // nombre d'éléments
        MPI_INT,            // type MPI
        right,              // rang du destinataire
        0,                  // tag (étiquette du message, doit matcher le Recv)
        MPI_COMM_WORLD      // communicateur
    );

    // Réception bloquante : attend jusqu'à réception du message
    MPI_Recv(
        &value,             // buffer où stocker le message reçu
        1,                  // nombre d'éléments attendus
        MPI_INT,            // type MPI
        left,               // rang de l'expéditeur attendu
        0,                  // tag (doit matcher le Send)
        MPI_COMM_WORLD,     // communicateur
        MPI_STATUS_IGNORE   // on ignore les infos de statut
    );

// ⚠️  ÉVITER LE DEADLOCK :
//     Rank 0 fait Send PUIS Recv  (envoie d'abord, boucle l'anneau en dernier)
//     Ranks 1..N font Recv PUIS Send  (reçoivent d'abord, puis transmettent)

// ❌  POURQUOI PAS MPI_Isend/MPI_Irecv ICI ?
//     → Non-bloquant = les envois partent tous en même temps = ordre non garanti
//     → On veut que la valeur circule SÉQUENTIELLEMENT dans l'ordre des rangs


// ╔══════════════════════════════════════════════════════════════╗
// ║  EXERCICE 12.3 — Dot product + norme (BLAS + MPI)           ║
// ╚══════════════════════════════════════════════════════════════╝

// COMMANDES CLÉS : MPI_Reduce (+ MPI_Allreduce si besoin du résultat partout)
// → Chaque processus calcule sa contribution LOCALE (avec BLAS)
// → MPI_Reduce combine toutes les contributions sur rank 0

    const int n = 1024;
    int loc_n = n / size;   // chunk local (n doit être divisible par size ici)

    // ... remplir x[] et y[] localement avec des valeurs aléatoires ...

    // Calcul local du dot product avec BLAS (ddot_) ou à la main :
    double loc_dot = 0.0;
    for (int i = 0; i < loc_n; i++) loc_dot += x[i] * y[i];

    // MPI_Reduce : additionne toutes les contributions locales → résultat sur rank 0
    double dot;
    MPI_Reduce(
        &loc_dot,           // valeur LOCALE à combiner (sendbuf)
        &dot,               // résultat final (recvbuf, valide SEULEMENT sur root)
        1,                  // nombre d'éléments
        MPI_DOUBLE,         // type MPI
        MPI_SUM,            // opération : MPI_SUM | MPI_MAX | MPI_MIN | MPI_PROD
        0,                  // rang du processus qui reçoit le résultat (root)
        MPI_COMM_WORLD      // communicateur
    );

    // Pour la norme : réduire la somme des carrés, puis sqrt sur rank 0
    double loc_norm_sq = 0.0;
    for (int i = 0; i < loc_n; i++) loc_norm_sq += x[i] * x[i];
    double norm_sq;
    MPI_Reduce(&loc_norm_sq, &norm_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Dot product = " << dot << endl;
        cout << "Norme de x  = " << sqrt(norm_sq) << endl;
    }

// ⚠️  NORME : ne pas réduire dnrm2_ directement ! Réduire les ||x_local||²,
//     puis faire sqrt() du total sur rank 0.

// Si tous les processus ont besoin du résultat (ex: pour la suite du calcul) :
    MPI_Allreduce(
        &loc_dot,           // valeur locale
        &dot,               // résultat disponible sur TOUS les processus
        1, MPI_DOUBLE, MPI_SUM,
        MPI_COMM_WORLD      // pas de 'root' ici
    );


// ╔══════════════════════════════════════════════════════════════╗
// ║  EXERCICE 12.4 — Intégrale trapèze (π/4)                    ║
// ╚══════════════════════════════════════════════════════════════╝

// COMMANDES CLÉS : MPI_Bcast + MPI_Reduce
// → Rank 0 lit n (nombre d'intervalles), le broadcast à tous
// → Chaque processus calcule sa portion de l'intégrale
// → MPI_Reduce somme les contributions → π sur rank 0

// ⚠️  CAS OÙ p NE DIVISE PAS n (IMPORTANT À L'EXAM) :
//     Distribuer les intervalles avec le pattern suivant :

    int n_intervals;
    if (rank == 0) cin >> n_intervals;
    MPI_Bcast(&n_intervals, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double h = 1.0 / n_intervals;   // largeur d'un intervalle

    // Chaque processus gère les intervalles : rank, rank+size, rank+2*size, ...
    // → Gère automatiquement le cas où n n'est pas divisible par size !
    double local_sum = 0.0;
    for (int i = rank; i < n_intervals; i += size) {
        double x_left  = i * h;
        double x_right = (i + 1) * h;
        local_sum += 0.5 * h * (f(x_left) + f(x_right));  // règle des trapèzes
    }

    double pi_approx;
    MPI_Reduce(&local_sum, &pi_approx, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) cout << "π ≈ " << 4.0 * pi_approx << endl;

// ⚠️  PATTERN "i += size" = distribution cyclique des intervalles
//     → Chaque rang prend les indices i = rank, rank+size, rank+2*size, ...
//     → Fonctionne même si n % size != 0 (certains rangs font juste moins d'itérations)


// ╔══════════════════════════════════════════════════════════════╗
// ║  EXERCICE 12.5 — Multiplication matrice-vecteur             ║
// ╚══════════════════════════════════════════════════════════════╝

// COMMANDES CLÉS : MPI_Scatter + MPI_Bcast + MPI_Gather
// → La matrice A est distribuée par lignes (chaque processus a loc_rows lignes)
// → Le vecteur x est broadcasté à tous (tout le monde en a besoin)
// → Chaque processus calcule sa portion de y = A*x (avec BLAS : dgemv_)
// → MPI_Gather collecte les portions de y sur rank 0

    int N = ...; // taille de la matrice (puissance de 2)
    int loc_rows = N / size;  // nombre de lignes locales

    // Distribuer les lignes de A depuis rank 0 vers tous
    MPI_Scatter(
        A_global,           // buffer source (sur rank 0 seulement)
        loc_rows * N,       // nombre d'éléments envoyés À CHAQUE processus
        MPI_DOUBLE,         // type
        A_local,            // buffer de réception local
        loc_rows * N,       // nombre d'éléments reçus
        MPI_DOUBLE,
        0,                  // root
        MPI_COMM_WORLD
    );

    // Broadcaster le vecteur x entier à tous les processus
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ... chaque processus calcule y_local = A_local * x (avec dgemv_ ou boucle) ...

    // Collecter les portions de y sur rank 0
    MPI_Gather(
        y_local,            // buffer source local
        loc_rows,           // nombre d'éléments envoyés par chaque processus
        MPI_DOUBLE,
        y_global,           // buffer de réception (sur rank 0 seulement)
        loc_rows,           // nombre d'éléments reçus de chaque processus
        MPI_DOUBLE,
        0,                  // root
        MPI_COMM_WORLD
    );

    if (rank == 0) { /* afficher y_global */ }

// ⚠️  SCATTER distribue des MORCEAUX DIFFÉRENTS à chaque processus (≠ Bcast)
// ⚠️  GATHER est l'INVERSE de Scatter : collecte les morceaux sur root


// ╔══════════════════════════════════════════════════════════════╗
// ║  EXERCICE 12.6 — Gradient conjugué parallèle                ║
// ╚══════════════════════════════════════════════════════════════╝

// COMMANDES CLÉS : MPI_Scatter + MPI_Allreduce + MPI_Gather (+ Bcast)
// → Même distribution que Ex 12.5 (A par lignes, x et b par morceaux)
// → À chaque itération du gradient conjugué, on a besoin de :
//     · Produits scalaires (dot products) → MPI_Allreduce (tous ont besoin du résidu)
//     · Multiplications matrice-vecteur  → comme Ex 12.5
//     · Mises à jour de vecteurs         → locales, pas de communication

    // Produit scalaire distribué (ex: r·r pour calculer le résidu) :
    double loc_rr = 0.0;
    for (int i = 0; i < loc_n; i++) loc_rr += r_local[i] * r_local[i];

    double rr;
    // Allreduce car TOUS les processus ont besoin de rr pour continuer l'algo
    MPI_Allreduce(&loc_rr, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Multiplication A*p distribuée : chaque processus calcule ses lignes,
    // mais a besoin du vecteur p ENTIER → MPI_Allgather (ou Bcast après Gather)
    // OU : utiliser MPI_Allgather pour que chaque proc reçoit toutes les portions de p

    MPI_Allgather(
        p_local,            // portion locale de p
        loc_n,              // taille de la portion locale
        MPI_DOUBLE,
        p_global,           // vecteur p complet (sur tous les processus)
        loc_n,
        MPI_DOUBLE,
        MPI_COMM_WORLD      // pas de root : tout le monde reçoit tout
    );

// ⚠️  Allreduce et Allgather = versions "All" → résultat sur TOUS les processus
//     → Nécessaire dans le gradient conjugué car chaque itération
//       a besoin des valeurs globales pour calculer alpha, beta, etc.


// ================================================================
//  RÉSUMÉ DES COMMANDES PAR EXERCICE
// ================================================================
//
//  Ex 12.1  →  MPI_Bcast
//               "rank 0 lit, tout le monde reçoit"
//
//  Ex 12.2  →  MPI_Send + MPI_Recv  (attention à l'ordre Send/Recv !)
//               "communication séquentielle point-à-point, éviter deadlock"
//
//  Ex 12.3  →  MPI_Reduce (ou MPI_Allreduce)
//               "chacun calcule localement, on additionne globalement"
//
//  Ex 12.4  →  MPI_Bcast + MPI_Reduce
//               "distribuer n, calculer localement, réduire le résultat"
//               Pattern i += size pour gérer n % size != 0
//
//  Ex 12.5  →  MPI_Scatter + MPI_Bcast + MPI_Gather
//               "distribuer la matrice, broadcaster le vecteur, collecter le résultat"
//
//  Ex 12.6  →  MPI_Scatter + MPI_Allreduce + MPI_Allgather + MPI_Gather
//               "gradient conjugué : les scalaires globaux à chaque itération"
//
// ================================================================
//  TYPES MPI COURANTS
// ================================================================
//  MPI_INT    → int
//  MPI_DOUBLE → double
//  MPI_FLOAT  → float
//
// ================================================================
//  OPÉRATIONS POUR MPI_Reduce / MPI_Allreduce
// ================================================================
//  MPI_SUM   → addition
//  MPI_MAX   → maximum
//  MPI_MIN   → minimum
//  MPI_PROD  → produit


// A*B en faisant ligne par ligne
#include <iostream>
#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <Accelerate/Accelerate.h>
using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 1024;
    int base_n = n / size;
    int rem    = n % size;
    int loc_n  = base_n + (rank == size-1 ? rem : 0);

    double* A_loc = new double[loc_n * n];
    double* B     = new double[n * n];
    double* C_loc = new double[loc_n * n]();

    srand(rank + 42);
    for (int i = 0; i < loc_n; i++)
        for (int j = 0; j < n; j++)
            A_loc[i*n+j] = (double)rand() / RAND_MAX;

    srand(0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            B[i*n+j] = (double)rand() / RAND_MAX;

    // Temps
    double t_start = MPI_Wtime();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                loc_n, n, n,
                1.0, A_loc, n, B, n, 0.0, C_loc, n);
    double t_end = MPI_Wtime();

    // Norme
    double local_norm_sq = cblas_ddot(loc_n * n, C_loc, 1, C_loc, 1);
    double global_norm_sq = 0.0;
    MPI_Reduce(&local_norm_sq, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Gatherv pour gérer le remainder
    int* recvcounts = new int[size];
    int* displs     = new int[size];
    for (int i = 0; i < size; i++) {
        recvcounts[i] = (i == size-1 ? base_n + rem : base_n) * n;
        displs[i]     = i * base_n * n;
    }

    double* C = (rank == 0) ? new double[n*n]() : nullptr;
    MPI_Gatherv(C_loc, loc_n*n, MPI_DOUBLE,
                C, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Frobenius norm: " << sqrt(global_norm_sq) << endl;
        cout << "Time (dgemm): "   << t_end - t_start << " s" << endl;
        delete[] C;
    }

    delete[] A_loc;
    delete[] B;
    delete[] C_loc;
    delete[] recvcounts;
    delete[] displs;

    MPI_Finalize();
    return 0;
}


// solve Ai xi = bi 
#include <iostream>
#include <mpi.h>
#include <cmath>
#include <cstdlib>
using namespace std;

#define F77NAME(x) x##_

extern "C" {
    // ── BLAS niveau 1 ─────────────────────────────────────────────────────
    double F77NAME(ddot)  (int* n, double* x, int* incx, double* y, int* incy);
    double F77NAME(dnrm2) (int* n, double* x, int* incx);
    void   F77NAME(dcopy) (int* n, double* x, int* incx, double* y, int* incy);
    void   F77NAME(dscal) (int* n, double* alpha, double* x, int* incx);
    void   F77NAME(daxpy) (int* n, double* alpha, double* x, int* incx,
                                                  double* y, int* incy);
    // ── BLAS niveau 2 ─────────────────────────────────────────────────────
    void F77NAME(dgemv)(char* trans, int* m, int* n,
                        double* alpha, double* A, int* lda,
                        double* x, int* incx,
                        double* beta,  double* y, int* incy);
    // ── BLAS niveau 3 ─────────────────────────────────────────────────────
    void F77NAME(dgemm)(char* transa, char* transb, int* m, int* n, int* k,
                        double* alpha, double* A, int* lda,
                                       double* B, int* ldb,
                        double* beta,  double* C, int* ldc);
    // ── LAPACK — systèmes linéaires ───────────────────────────────────────
    void F77NAME(dgesv) (int* n, int* nrhs, double* A, int* lda, int* ipiv,
                         double* b, int* ldb, int* info);
    void F77NAME(dpotrf)(char* uplo, int* n, double* A, int* lda, int* info);
    void F77NAME(dpotrs)(char* uplo, int* n, int* nrhs, double* A, int* lda,
                         double* b, int* ldb, int* info);
    // ── LAPACK — valeurs propres ──────────────────────────────────────────
    void F77NAME(dsyev)(char* jobz, char* uplo, int* n, double* A, int* lda,
                        double* w, double* work, int* lwork, int* info);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 8;
    srand(rank + 1);

    // ── ÉTAPE 1 : Construire A = M*M^T + n*I ─────────────────────────────
    double* M_local = new double[n*n];
    double* A_local = new double[n*n]();

    for (int i = 0; i < n*n; i++)
        M_local[i] = ((double)rand() / RAND_MAX) - 0.5;

    int N     = n;
    int NN    = n*n;
    int INC   = 1;
    double alpha = 1.0, beta = 0.0;
    char N_char = 'N', T_char = 'T';

    // A = M * M^T
    F77NAME(dgemm)(&N_char, &T_char, &N, &N, &N,
                   &alpha, M_local, &N,
                           M_local, &N,
                   &beta,  A_local, &N);

    // A += n*I
    for (int i = 0; i < n; i++)
        A_local[i*n + i] += n;

    // Sauvegarder A avant que dgesv l'écrase
    double* A_orig = new double[n*n];
    F77NAME(dcopy)(&NN, A_local, &INC, A_orig, &INC);

    // ── ÉTAPE 2 : Construire b ────────────────────────────────────────────
    double* b_local = new double[n];
    for (int i = 0; i < n; i++)
        b_local[i] = (double)rand() / RAND_MAX;

    double* b_orig = new double[n];
    F77NAME(dcopy)(&N, b_local, &INC, b_orig, &INC);

    // ── ÉTAPE 3 : Résoudre A*x = b avec dgesv ────────────────────────────
    int NRHS = 1;
    int INFO = 0;
    int* ipiv = new int[n]();

    F77NAME(dgesv)(&N, &NRHS, A_local, &N, ipiv, b_local, &N, &INFO);

    if (INFO != 0) {
        cerr << "Rank " << rank << ": dgesv failed INFO=" << INFO << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // b_local contient maintenant x

    // ── ÉTAPE 4 : Résidu r = b_orig - A_orig * x ─────────────────────────
    double* resid = new double[n];
    F77NAME(dcopy)(&N, b_orig, &INC, resid, &INC);   // resid = b_orig

    double alpha2 = -1.0, beta2 = 1.0;
    F77NAME(dgemv)(&N_char, &N, &N,
                   &alpha2, A_orig,  &N,
                            b_local, &INC,
                   &beta2,  resid,   &INC);   // resid = b - A*x

    double nrm     = F77NAME(dnrm2)(&N, resid, &INC);
    double local_r = nrm * nrm;

    // ── ÉTAPE 5 : Réduction et affichage ─────────────────────────────────
    double global_R = 0.0;
    MPI_Reduce(&local_r, &global_R, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "R       = " << global_R       << endl;
        cout << "sqrt(R) = " << sqrt(global_R) << endl;
    }

    delete[] M_local;
    delete[] A_local;
    delete[] A_orig;
    delete[] b_local;
    delete[] b_orig;
    delete[] resid;
    delete[] ipiv;

    MPI_Finalize();
    return 0;
}



#include <iostream>
#include <stdexcept>
#include <algorithm>
using namespace std;
#define F77NAME(x) x##_

extern "C" {
    // Factorisation LU de A : A → P*L*U, ipiv contient les permutations
    void F77NAME(dgetrf)(const int& m,
                         const int& n,
                         double* A,
                         const int& lda,
                         int* ipiv,
                         int& info);

    // Inversion de A à partir de sa factorisation LU
    void F77NAME(dgetri)(const int& n,
                         double* A,
                         const int& lda,
                         int* ipiv,
                         double* work,
                         const int& lwork,
                         int& info);
}

// Retourne une nouvelle matrice A^-1, sans modifier A
// La matrice est stockée en colonne-major (format Fortran) : A[i][j] = A[j*n + i]
// L'appelant est responsable du delete[] du pointeur retourné
double* inverse(const double* A, int n) {
    // Copie de A car dgetrf/dgetri travaillent in-place
    double* Ainv = new double[n * n];
    copy(A, A + n * n, Ainv);

    int* ipiv = new int[n];
    int info;

    // Étape 1 : factorisation LU
    F77NAME(dgetrf)(n, n, Ainv, n, ipiv, info);
    if (info != 0) throw runtime_error("dgetrf failed");

    // Étape 2 : calcul de la taille optimale du workspace (appel avec lwork=-1)
    int lwork = -1;
    double wkopt;
    F77NAME(dgetri)(n, Ainv, n, ipiv, &wkopt, lwork, info);
    lwork = (int)wkopt;

    double* work = new double[lwork];

    // Étape 3 : inversion effective
    F77NAME(dgetri)(n, Ainv, n, ipiv, work, lwork, info);
    if (info != 0) throw runtime_error("dgetri failed");

    delete[] ipiv;
    delete[] work;

    return Ainv;
}


// Utilisation 
    double* Ainv = inverse(A, n);

    // Affichage de A^-1 en column-major
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            cout << Ainv[j * n + i] << "\t";
        cout << "\n";
    }

    delete[] Ainv;  // ne pas oublier !
    return 0;


// Remplir matrice en column-major
double* A = new double[n * n]();  // init à 0

for (int j = 0; j < n; j++)
    for (int i = 0; i < n; i++) {

        // Diagonales : i - j = constante
        if (i == j)      A[j * n + i] = 3.0;   // diag principale  (offset 0)
        if (i == j + 1)  A[j * n + i] = -1.0;  // diag inférieure  (offset -1)
        if (i == j - 1)  A[j * n + i] = 2.0;   // diag supérieure  (offset +1)
        if (i == j + 2)  A[j * n + i] = 5.0;   // 2ème diag inf    (offset -2)
        if (i == j - 2)  A[j * n + i] = 5.0;   // 2ème diag sup    (offset +2)

        // Première ligne : i == 0
        if (i == 0)      A[j * n + i] = 9.0;

        // Dernière ligne : i == n-1
        if (i == n - 1)  A[j * n + i] = 7.0;

        // Première colonne : j == 0
        if (j == 0)      A[j * n + i] = 4.0;

        // Dernière colonne : j == n-1
        if (j == n - 1)  A[j * n + i] = 6.0;

        // Coin particulier
        if (i == 0 && j == n - 1)  A[j * n + i] = 99.0;
    }

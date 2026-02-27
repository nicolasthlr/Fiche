g++ main.cpp -llapack -lblas

// Générer un double entre 0 et 1
double x = (double)rand() / RAND_MAX;

#include <iostream>
#include <cstdlib>
#include <cmath>
#include "cblas.h"
using namespace std;

// Cblas (replace "d" by "z" if we deal with complex)

// y <-- ax+y
cblas_daxpy(n,a,x,inc x,y,inc y);

// y <-- x
cblas_dcopy(n,x,inc x,y,inc y);

// x . y
cblas_ddot(n,x,inc x, y, inc y);

// Euclidian norm || x ||2
cblas_dnrm2(n,x,inc x);

// x <-- ax
cblas_dscal(n,a,x,inc x);

// y <-- aAx + by
// Order = CblasRowMajor or CblasColMajor, TransA = CblasNoTrans or CblasTrans, lda = nb colonnes (en row major) 
cblas_dgemv(Order,TransA,M,N,a,A,lda,x,inc x,b,y,inc y);

// C <-- aAB+bC
cblas_dgemm(Order,TransA,TransB,M,N,K,a,A,lda,B,ldb,b,C,ldc);


// ================================================================
//         FICHE RÉVISION LAPACK — Fonctions essentielles
// ================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>

// ================================================================
// CONVENTION OBLIGATOIRE — Toujours mettre en haut du fichier
// ================================================================

// Macro qui ajoute automatiquement le _ à la fin du nom de fonction
// (convention Fortran, obligatoire pour appeler LAPACK depuis C++)
#define F77NAME(x) x##_

// Toutes les routines LAPACK doivent être déclarées en extern "C"
// pour éviter le name mangling du C++

// Compiler avec : mpicxx main.cpp -o main -llapack -lblas


// ================================================================
// RÈGLES GÉNÉRALES LAPACK
// ================================================================

// 1. Tous les arguments sont passés PAR RÉFÉRENCE (const int& n, pas int n)
// 2. Les matrices sont stockées en 1D, COLONNE PAR COLONNE (column-major)
//    → A[i][j] en 2D = A[i + j*n] en 1D   (ordre Fortran)
// 3. Les matrices sont souvent ÉCRASÉES après l'appel
//    → faire une copie si on en a besoin après
// 4. Toujours vérifier info après l'appel :
//    info = 0  → succès
//    info < 0  → argument invalide (le -info ème argument)
//    info > 0  → échec numérique (ex: matrice singulière)

// Nommage des routines :
// d = double   (s = float, z = complex double)
// ge = matrice générale   (sy = symétrique, tr = triangulaire)
// sv = solve   (ev = eigenvalues, trf = factorisation)


// ================================================================
// 1. SYSTÈME LINÉAIRE Ax = b  →  dgesv_
// ================================================================

extern "C" {
    void F77NAME(dgesv)(const int& n,     // taille de la matrice A (n×n)
                        const int& nrhs,  // nombre de seconds membres (colonnes de b)
                        double* A,        // matrice n×n → ÉCRASÉE par la facto LU
                        const int& lda,   // leading dimension = n
                        int* ipiv,        // tableau de n entiers (pivot) → allouer n ints
                        double* b,        // second membre → ÉCRASÉ par la solution x
                        const int& ldb,   // leading dimension de b = n
                        int& info);       // 0=succès, <0=arg invalide, >0=singulier
}

// EXEMPLE D'UTILISATION :
//
//    const int n = 4;
//    const int nrhs = 1;    // un seul second membre
//    double A[n*n] = { ... };
//    double b[n]   = { ... };
//    int ipiv[n];
//    int info;
//
//    F77NAME(dgesv)(n, nrhs, A, n, ipiv, b, n, info);
//
//    if (info != 0) printf("Erreur dgesv : info = %d\n", info);
//    // b contient maintenant la solution x


// ================================================================
// 2. VALEURS PROPRES d'une matrice SYMÉTRIQUE  →  dsyev_
// ================================================================

extern "C" {
    void F77NAME(dsyev)(const char& jobz,  // 'N' = valeurs propres seulement
                                           // 'V' = valeurs propres + vecteurs propres
                        const char& uplo,  // 'U' = triangle supérieur stocké
                                           // 'L' = triangle inférieur stocké
                        const int& n,      // taille de la matrice
                        double* a,         // matrice n×n → ÉCRASÉE par les vecteurs propres si jobz='V'
                        const int& lda,    // leading dimension = n
                        double* w,         // tableau de n doubles → valeurs propres en sortie (ordre croissant)
                        double* work,      // workspace temporaire
                        const int& lwork,  // taille du workspace (-1 pour query)
                        int* info);        // 0=succès, >0=non convergence
}

// ⚠️  TOUJOURS APPELER EN DEUX FOIS :

// ÉTAPE 1 : demander la taille optimale du workspace (lwork = -1)
//    char jobz = 'N', uplo = 'U';
//    int lwork = -1;
//    double work_query;
//    int info;
//    double w[n];
//
//    F77NAME(dsyev)(jobz, uplo, n, A, n, w, &work_query, lwork, &info);
//
// ÉTAPE 2 : allouer le workspace et faire le vrai calcul
//    lwork = (int)work_query;
//    double* work = new double[lwork];
//
//    F77NAME(dsyev)(jobz, uplo, n, A, n, w, work, lwork, &info);
//
//    // w[0], w[1], ..., w[n-1] contiennent les valeurs propres (ordre croissant)
//    delete[] work;

// CALCULER LE DÉTERMINANT depuis les valeurs propres :
//    double det = 1.0;
//    for (int k = 0; k < n; k++) det *= w[k];
//    // det(A) = produit de toutes les valeurs propres


// ================================================================
// 3. FACTORISATION LU  →  dgetrf_
// ================================================================

extern "C" {
    void F77NAME(dgetrf)(const int& m,   // nombre de lignes de A
                         const int& n,   // nombre de colonnes de A
                         double* a,      // matrice → ÉCRASÉE par la factorisation L et U
                         const int& lda, // leading dimension = m
                         int* ipiv,      // tableau de min(m,n) entiers (pivots)
                         int& info);     // 0=succès, >0=matrice singulière
}

// EXEMPLE D'UTILISATION :
//
//    int ipiv[n];
//    int info;
//    F77NAME(dgetrf)(n, n, A, n, ipiv, info);
//
//    if (info != 0) printf("Erreur dgetrf : info = %d\n", info);
//    // A contient maintenant L et U (combinées)

// ⚠️  dgetrf_ seul ne résout rien — il factorise seulement.
//     Pour résoudre ensuite, utiliser dgesv_ directement
//     (qui fait la factorisation LU + résolution en un seul appel).


// ================================================================
// 4. RÉSUMÉ DES PARAMÈTRES COURANTS
// ================================================================

//  Paramètre   Valeur courante     Signification
//  ---------   ---------------     -------------
//  n           taille du problème  nombre de lignes/colonnes
//  lda         n                   leading dimension (= n pour matrices carrées)
//  ldb         n                   leading dimension du second membre
//  nrhs        1                   un seul vecteur b (second membre)
//  jobz        'N'                 valeurs propres seulement (pas les vecteurs)
//  uplo        'U'                 triangle supérieur (pour matrices symétriques)
//  lwork       -1 puis optimal     taille du workspace (query puis allocation)
//  info        0                   succès (toujours vérifier !)
//  ipiv        int[n]              tableau de pivots (toujours allouer !)


// ================================================================
// 5. STOCKAGE COLUMN-MAJOR (ORDRE FORTRAN) — IMPORTANT
// ================================================================

// LAPACK stocke les matrices COLONNE PAR COLONNE :
//
//  Matrice 3×3 :   | 1 2 3 |
//                  | 4 5 6 |
//                  | 3 8 9 |
//
//  En mémoire :    [1, 4, 3, 2, 5, 8, 3, 6, 9]
//                   col0      col1      col2
//
//  Formule :  A[i][j]  →  A[i + j*n]   (i=ligne, j=colonne)
//
// ⚠️  En C++, les tableaux 2D sont stockés LIGNE PAR LIGNE (row-major)
//     → A[i][j] en C++ = A[i*n + j]
//     → Il faut donc TRANSPOSER si on remplit la matrice en C++ style
//     → OU remplir directement en column-major : A[i + j*n]

// Pour une matrice SYMÉTRIQUE : peu importe car A = A^T
// → on peut remplir en row-major et passer uplo='U', ça revient au même


// ================================================================
// 6. UTILISATION AVEC MPI — PATTERN TYPE DE L'EXAMEN
// ================================================================

// Chaque rank calcule quelque chose localement avec LAPACK,
// puis MPI combine les résultats.

// PATTERN EXAM (comme le test det(A)) :
//
//    // 1. Chaque rank remplit et traite son bloc localement
//    double A_local[n*n];
//    // ... remplir A_local ...
//
//    // 2. LAPACK calcule les valeurs propres localement
//    double w[n];
//    // ... dsyev_ ...
//
//    // 3. Calcul local du déterminant
//    double loc_det = 1.0;
//    for (int k = 0; k < n; k++) loc_det *= w[k];
//
//    // 4. MPI combine les résultats
//    double det_global;
//    MPI_Reduce(&loc_det, &det_global, 1, MPI_DOUBLE, MPI_PROD, 0, MPI_COMM_WORLD);
//    //                                                ^^^^^^^^
//    //                                    MPI_PROD pour un produit (det = produit des det locaux)
//    //                                    MPI_SUM  pour une somme  (dot product, intégrale...)
//
//    if (rank == 0) printf("det(A) = %e\n", det_global);


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






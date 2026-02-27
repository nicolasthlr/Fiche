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


#define F77NAME(x) x##_

// Solve Ax = b
// On input b contains RHS vector, on output b contains solution
extern "C" {
  void F77NAME(dgesv)(const int& n, const int& nrhs, const double * A,
                      const int& lda, int * ipiv, double * b,
                      const int& ldb, int& info);
}

// Eigenvalues/eigenvectors of symmetric matrix
extern "C" {
  void F77NAME(dsyev)(const char& v, const char& ul, const int& n,
                      double* a, const int& lda, double* w,
                      double* work, const int& lwork, int* info);
}

// Decomposition LU 
extern "C" {
  void F77NAME(dgetrf)(const int& m, const int& n, double* a, 
                       const int& lda, int* ipiv, int& info);
}


// Example
int n = 50; // Problem size
int nrhs = 1; // Number of RHS vectors
int info = 0;
double* A = new double[n*n];
int* ipiv = new int[n]; // Vector for pivots
double* b = new double[n]; // RHS vector / output vector





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






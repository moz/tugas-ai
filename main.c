#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
//---------------------------------------------------------------------------
#define ROOT 0

#define MESSAGE_TAG 1000
#define WORK_TAG    1001
//---------------------------------------------------------------------------
#define SWARM_WEIGHT 1.0f//2.5f
#define SELF_WEIGHT 1.0f//1.5f
#define GENERATIONS 100

#define N_MIN -20
#define N_MAX 20
#define M_MIN -20
#define M_MAX 20
#define E_MIN 0.0f
#define E_MAX 1.0f
#define EB_MIN 0.0f
#define EB_MAX 1.0f

#define VEL_MAX 0.1f
#define VEL_MIN -0.1f
//---------------------------------------------------------------------------
typedef struct Position {
    int n;
    int m;
    float e;
    float eb;
} Position;

typedef struct Velocity {
    float n;
    float m;
    float e;
    float eb;
} Velocity;

typedef struct Particle {
    struct Position pos;
    struct Velocity velocity;

    float fitness;
} Particle;
//---------------------------------------------------------------------------
// Algorithm Functions
float fitness(Particle *f);
float randFloat(float max, float min);
int randInt(int min, int max);
void printParticle(Particle p, char *c, int id, int g);
void copyParticle(Particle *s, Particle *d);
//---------------------------------------------------------------------------
// MPI Functions
void getStatus(MPI_Status status, char* pcStatus);
void sendParticle(Particle *p, int destination);
void recvParticle(Particle *p, int source);
void bcastParticle(Particle *p, int source);
//---------------------------------------------------------------------------
main (int argc, char** argv)
{
    int iNameLen;
    int iSwarmSize;

    Particle particle;
    int id;
  
    char* pcName    = (char*)malloc(1024);
    char* pcMessage = (char*)malloc(1024);
    char* pcStatus  = (char*)malloc(1024);

    MPI_Init(&argc, &argv);
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &iSwarmSize);
    MPI_Get_processor_name(pcName, &iNameLen);

    srand(time(NULL) + (id * 1984));
  
    int bRoot = (id == ROOT);

    particle.pos.n = randInt(N_MIN, N_MAX);
    particle.pos.m = randInt(M_MIN, M_MAX);
    particle.pos.e = randFloat(E_MAX, E_MIN);
    particle.pos.eb = randFloat(EB_MAX, EB_MIN);
    particle.velocity.n = randFloat(1, 2);
    particle.velocity.m = randFloat(1, 2);
    particle.velocity.e = randFloat(0.1, 0.7);
    particle.velocity.eb = randFloat(0.1, 0.7);

    if(particle.velocity.n == 0) particle.velocity.n = 1;
    if(particle.velocity.m == 0) particle.velocity.m = 1;

    if (!bRoot) {
        sprintf(pcMessage, "%i, %s", id, pcName);
        MPI_Send(pcMessage, strlen(pcMessage)+1, MPI_CHAR, ROOT, MESSAGE_TAG, MPI_COMM_WORLD);
    } else {
        printf("Initializing Nodes\n");
        printf("*********************************************************\n");
        printf("Root Node: %s\n", pcName);
        printf("Awaiting worker node responses...\n");
        int iSource;
        for (iSource = 1; iSource < iSwarmSize; iSource++) {
            MPI_Recv(pcMessage, 1024, MPI_CHAR, iSource, MESSAGE_TAG, MPI_COMM_WORLD, &status);
            getStatus(status, pcStatus);
            printf("AgentNode: %s\t| Status: %s\n", pcMessage, pcStatus);
        }
        printf("Node initialization complete...\n");
        printf("%i Nodes Registered for work...\n", iSwarmSize);
        printf("*********************************************************\n");
    }

    //int iSwarmBest = 0;
    Particle pSwarmBest;
    Particle best;

    fitness(&particle);
    copyParticle(&particle, &best);
    //printParticle(particle, "current", id, -1);

    if (bRoot) {
        copyParticle(&particle, &pSwarmBest);

        Particle tmpParticle;
        int iParticle;
        for (iParticle = 1; iParticle < iSwarmSize; iParticle++) {
            recvParticle(&tmpParticle, iParticle);
            if (tmpParticle.fitness > pSwarmBest.fitness) {
                copyParticle(&tmpParticle, &pSwarmBest);
            }
        }
        printParticle(pSwarmBest, "swarm best", id, -1);
    } else {
        sendParticle(&particle, ROOT);
    }  

    bcastParticle(&pSwarmBest, ROOT);

    MPI_Barrier(MPI_COMM_WORLD);

    int iGeneration;
    for (iGeneration = 0; iGeneration < GENERATIONS; iGeneration++) {
        if (bRoot) {
            printf("Generation-> %i\n", iGeneration);
        }
   
        float r1 = randFloat(0, 1); 
        float r2  = randFloat(0, 1);
/*
        fNewVel = particle.velocity.x + 
                  SELF_WEIGHT  * (particle.best_pos.x - particle.pos.x)  * fSelfRand + 
                  SWARM_WEIGHT * (pSwarmBest.best_pos.x - particle.pos.x) * fSwarmRand;
*/
        particle.velocity.n = particle.velocity.n
                            + SELF_WEIGHT * r1 * (best.pos.n - particle.pos.n)
                            + SWARM_WEIGHT * r2 * (pSwarmBest.pos.n - particle.pos.n);

        particle.velocity.m = particle.velocity.m
                            + SELF_WEIGHT * r1 * (best.pos.m - particle.pos.m)
                            + SWARM_WEIGHT * r2 * (pSwarmBest.pos.m - particle.pos.m);

        particle.velocity.e = particle.velocity.e
                            + SELF_WEIGHT * r1 * (best.pos.e - particle.pos.e)
                            + SWARM_WEIGHT * r2 * (pSwarmBest.pos.e - particle.pos.e);

        particle.velocity.eb = particle.velocity.eb
                             + SELF_WEIGHT * r1 * (best.pos.eb - particle.pos.eb)
                             + SWARM_WEIGHT * r2 * (pSwarmBest.pos.eb - particle.pos.eb);

        particle.pos.n += (int)particle.velocity.n;
        particle.pos.m += (int)particle.velocity.m;
        particle.pos.e += particle.velocity.e;
        particle.pos.eb += particle.velocity.eb;

        if(particle.pos.n > N_MAX) particle.pos.n = N_MAX;
        if(particle.pos.n < N_MIN) particle.pos.n = N_MIN;
        if(particle.pos.m > M_MAX) particle.pos.m = M_MAX;
        if(particle.pos.m < M_MIN) particle.pos.m = M_MIN;
        if(particle.pos.e > E_MAX) particle.pos.e = E_MAX;
        if(particle.pos.e < E_MIN) particle.pos.e = E_MIN;
        if(particle.pos.eb > EB_MAX) particle.pos.eb = EB_MAX;
        if(particle.pos.eb < EB_MIN) particle.pos.eb = EB_MIN;

        fitness(&particle);
        //printf("%d %d", id, iGeneration);
        //printParticle(particle, "current", id, iGeneration);

        if(particle.fitness > best.fitness) {
            copyParticle(&particle, &best);
        }

        if (bRoot) {
            if(particle.fitness > pSwarmBest.fitness) {
                copyParticle(&particle, &pSwarmBest);
            }

            Particle tmpParticle;
            int iParticle;
            for (iParticle = 1; iParticle < iSwarmSize; iParticle++) {
                recvParticle(&tmpParticle, iParticle);

                if (tmpParticle.fitness > pSwarmBest.fitness) {
                    copyParticle(&tmpParticle, &pSwarmBest);
                }
            }
        } else {
            sendParticle(&particle, ROOT);
        }

        bcastParticle(&pSwarmBest, ROOT);

        if (bRoot) {
            //printf("SwarmBest: ");
            printParticle(pSwarmBest, "swarm best", id, iGeneration);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (bRoot) {
        printf("And the winner is: (%d, %d, %.3f, %.3f) -> %.3f\n",  pSwarmBest.pos.n, pSwarmBest.pos.m, pSwarmBest.pos.e, pSwarmBest.pos.eb, pSwarmBest.fitness);
    }
 
    MPI_Finalize();
  
    return 0;
}
//-------------------------------------------------
float randFloat(float a, float b)
{
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r; 
}

int randInt(int min, int max) {
    return (rand() % (max-min)) + min;
}
//-------------------------------------------------
float fitness(Particle *f)
{
    f->fitness = 0;
    f->fitness = -1*pow(f->pos.e, 2);
    f->fitness += -1*pow(f->pos.eb, 2);
    f->fitness += -1*pow(f->pos.n, 2);
    f->fitness += -1*pow(f->pos.m, 2);
    //f.fitness =  -1 * pow(x - 0.5, 3) + x;
    //f.fitness =  -1 * pow(x - 0.5, 3) + x - pow(y, 2) - (x * y);

    return f->fitness;
}
//-------------------------------------------------
void sendParticle(Particle *p, int destination)
{
    MPI_Send(&(p->fitness), 1, MPI_FLOAT, destination, 100, MPI_COMM_WORLD);

    MPI_Send(&(p->pos.n), 1, MPI_INT, destination, 101, MPI_COMM_WORLD);
    MPI_Send(&(p->pos.m), 1, MPI_INT, destination, 102, MPI_COMM_WORLD);
    MPI_Send(&(p->pos.e), 1, MPI_FLOAT, destination, 103, MPI_COMM_WORLD);
    MPI_Send(&(p->pos.eb), 1, MPI_FLOAT, destination, 104, MPI_COMM_WORLD);

    MPI_Send(&(p->velocity.n), 1, MPI_FLOAT, destination, 105, MPI_COMM_WORLD);
    MPI_Send(&(p->velocity.m), 1, MPI_FLOAT, destination, 106, MPI_COMM_WORLD);
    MPI_Send(&(p->velocity.e), 1, MPI_FLOAT, destination, 107, MPI_COMM_WORLD);
    MPI_Send(&(p->velocity.eb), 1, MPI_FLOAT, destination, 108, MPI_COMM_WORLD);
}

void recvParticle(Particle *p, int source)
{
    MPI_Status status;
    MPI_Recv(&(p->fitness), 1, MPI_FLOAT, source, 100,  MPI_COMM_WORLD, &status);

    MPI_Recv(&(p->pos.n), 1, MPI_INT, source, 101, MPI_COMM_WORLD, &status);
    MPI_Recv(&(p->pos.m), 1, MPI_INT, source, 102, MPI_COMM_WORLD, &status);
    MPI_Recv(&(p->pos.e), 1, MPI_FLOAT, source, 103, MPI_COMM_WORLD, &status);
    MPI_Recv(&(p->pos.eb), 1, MPI_FLOAT, source, 104, MPI_COMM_WORLD, &status);

    MPI_Recv(&(p->velocity.n), 1, MPI_FLOAT, source, 105, MPI_COMM_WORLD, &status);
    MPI_Recv(&(p->velocity.m), 1, MPI_FLOAT, source, 106, MPI_COMM_WORLD, &status);
    MPI_Recv(&(p->velocity.e), 1, MPI_FLOAT, source, 107, MPI_COMM_WORLD, &status);
    MPI_Recv(&(p->velocity.eb), 1, MPI_FLOAT, source, 108, MPI_COMM_WORLD, &status);
}

void bcastParticle(Particle *p, int source)
{
    MPI_Bcast(&(p->fitness), 1, MPI_FLOAT, source, MPI_COMM_WORLD);

    MPI_Bcast(&(p->pos.n), 1, MPI_INT, source, MPI_COMM_WORLD);
    MPI_Bcast(&(p->pos.m), 1, MPI_INT, source, MPI_COMM_WORLD);
    MPI_Bcast(&(p->pos.e), 1, MPI_FLOAT, source, MPI_COMM_WORLD);
    MPI_Bcast(&(p->pos.eb), 1, MPI_FLOAT, source, MPI_COMM_WORLD);

    MPI_Bcast(&(p->velocity.n), 1, MPI_FLOAT, source, MPI_COMM_WORLD);
    MPI_Bcast(&(p->velocity.m), 1, MPI_FLOAT, source, MPI_COMM_WORLD);
    MPI_Bcast(&(p->velocity.e), 1, MPI_FLOAT, source, MPI_COMM_WORLD);
    MPI_Bcast(&(p->velocity.eb), 1, MPI_FLOAT, source, MPI_COMM_WORLD);
}

//-------------------------------------------------
void printParticle(struct Particle p, char *c, int id, int g)
{
    printf("[%s %02d %03d](%d %d %.3f %.3f) (%.3f %.3f %.3f %.3f) :: %.3f\n", c, id, g, p.pos.n, p.pos.m, p.pos.e, p.pos.eb, p.velocity.n, p.velocity.m, p.velocity.e, p.velocity.eb, p.fitness);
}

void copyParticle(Particle *s, Particle *d) {
    d->fitness = s->fitness;

    d->pos.n = s->pos.n;
    d->pos.m = s->pos.m;
    d->pos.e = s->pos.e;
    d->pos.eb = s->pos.eb;

    d->velocity.n = s->velocity.n;
    d->velocity.m = s->velocity.m;
    d->velocity.e = s->velocity.e;
    d->velocity.eb = s->velocity.eb;
}
//-------------------------------------------------
void getStatus(MPI_Status status, char* pcStatus)
{
switch (status.MPI_ERROR) {
	case MPI_SUCCESS:
		sprintf(pcStatus, "Success");
		break;
	case MPI_ERR_BUFFER:
		sprintf(pcStatus, "Buffer Error");
		break;
	case MPI_ERR_COUNT:
		sprintf(pcStatus, "Count Error");
		break;
	case MPI_ERR_TYPE:
		sprintf(pcStatus, "Type Error");
		break;
	case MPI_ERR_TAG:
		sprintf(pcStatus, "Tag Error");
		break;
	case MPI_ERR_COMM:
		sprintf(pcStatus, "COMM Error");
		break;
	case MPI_ERR_RANK:
		sprintf(pcStatus, "Rank Error");
		break;
	case MPI_ERR_REQUEST:
		sprintf(pcStatus, "Request Error");
		break;
	case MPI_ERR_ROOT:
		sprintf(pcStatus, "Root Error");
		break;
	case MPI_ERR_GROUP:
		sprintf(pcStatus, "Group Error");
		break;	
	case MPI_ERR_OP:
		sprintf(pcStatus, "OP Error");
		break;
	case MPI_ERR_TOPOLOGY:
		sprintf(pcStatus, "Topology Error");
		break;
	case MPI_ERR_DIMS:
		sprintf(pcStatus, "DIMS Error");
		break;
	case MPI_ERR_ARG:
		sprintf(pcStatus, "Args Error");
		break;
	case MPI_ERR_UNKNOWN:
		sprintf(pcStatus, "Unknown Error");
		break;
	case MPI_ERR_TRUNCATE:
		sprintf(pcStatus, "Truncate Error");
		break;	
	case MPI_ERR_OTHER:
		sprintf(pcStatus, "Other Error");
		break;	
	case MPI_ERR_INTERN:
		sprintf(pcStatus, "Internal Error");
		break;
	case MPI_ERR_IN_STATUS:
		sprintf(pcStatus, "In Status Error");
		break;
	case MPI_ERR_PENDING:
		sprintf(pcStatus, "Pending Error");
		break;
	default:
		sprintf(pcStatus, "Unknown Error");
  }
}
//------------------------------------------------


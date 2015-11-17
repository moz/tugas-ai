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
#define SWARM_WEIGHT 2.5f
#define SELF_WEIGHT 1.5f
#define GENERATIONS 100 

#define N_MIN 1
#define N_MAX 20
#define M_MIN 1
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

typedef struct Particle {
    struct Position current_pos;
    struct Position velocity;

    float current_fitness;
} Particle;
//---------------------------------------------------------------------------
// Algorithm Functions
float fitness(Particle *f, int bSetBest);
float randFloat(float max, float min);
int randInt(int min, int max);
void  printParticle(Particle p);
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

    srand(time(NULL));

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &iSwarmSize);
    MPI_Get_processor_name(pcName, &iNameLen);
  
    int bRoot = (id == ROOT);

    particle.current_pos.n = randInt(N_MIN, N_MAX);
    particle.current_pos.m = randInt(M_MIN, M_MAX);
    particle.current_pos.e = randFloat(E_MAX, E_MIN);
    particle.current_pos.eb = randFloat(EB_MAX, EB_MIN);
    particle.velocity.n = randInt(-1, 1);
    particle.velocity.m = randInt(-1, 1);
    particle.velocity.e = randFloat(0.0, 0.5);
    particle.velocity.eb = randFloat(0.0, 0.5);
 
    fitness(&particle, 1);
  
    if (!bRoot) {
        sprintf(pcMessage, "%i, %s", id, pcName);
        MPI_Send(pcMessage, strlen(pcMessage)+1, MPI_CHAR, ROOT, MESSAGE_TAG, MPI_COMM_WORLD);
    } else {
        printf("Initializing Nodes\n");
        printf("*********************************************************\n");
        printf("Root Node: %s\n", pcName);
        printf("Awaiting worker node responses...\n");
        for (int iSource = 1; iSource < iSwarmSize; iSource++) {
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
  
    if (bRoot) {
        pSwarmBest.current_fitness = particle.current_fitness;
        pSwarmBest.best_pos.x   = particle.current_pos.x;
        pSwarmBest.best_pos.y   = particle.current_pos.y;

        Particle tmpParticle;
        for (int iParticle = 1; iParticle < iSwarmSize; iParticle++) {
            MPI_Recv(&tmpParticle.current_fitness, 1, MPI_FLOAT, iParticle, MESSAGE_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&tmpParticle.current_pos.x,   1, MPI_FLOAT, iParticle, MESSAGE_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&tmpParticle.current_pos.y,   1, MPI_FLOAT, iParticle, MESSAGE_TAG, MPI_COMM_WORLD, &status);
            if (tmpParticle.current_fitness > pSwarmBest.best_fitness) {
                //iSwarmBest              = iParticle;
                pSwarmBest.best_fitness = tmpParticle.current_fitness;
                pSwarmBest.best_pos.x   = tmpParticle.current_pos.x;
                pSwarmBest.best_pos.y   = tmpParticle.current_pos.y;
            }
        }
        printParticle(pSwarmBest);
    } else {
        MPI_Send(&particle.current_fitness, 1, MPI_FLOAT, ROOT, MESSAGE_TAG, MPI_COMM_WORLD);
        MPI_Send(&particle.current_pos.x,   1, MPI_FLOAT, ROOT, MESSAGE_TAG, MPI_COMM_WORLD);
        MPI_Send(&particle.current_pos.y,   1, MPI_FLOAT, ROOT, MESSAGE_TAG, MPI_COMM_WORLD);
    }  

    MPI_Bcast(&pSwarmBest.best_fitness, 1, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&pSwarmBest.best_pos.x,   1, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&pSwarmBest.best_pos.y,   1, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

  for (int iGeneration = 0; iGeneration < GENERATIONS; iGeneration++) {
    if (bRoot) {
      printf("Generation-> %i\n", iGeneration);
    }
   
    float fSwarmRand = randFloat(0, 1); 
    float fSelfRand  = randFloat(0, 1);
    float fNewVel;
    fNewVel = particle.velocity.x + 
              SELF_WEIGHT  * (particle.best_pos.x - particle.current_pos.x)  * fSelfRand + 
              SWARM_WEIGHT * (pSwarmBest.best_pos.x - particle.current_pos.x) * fSwarmRand;

    if (fNewVel > VEL_MAX) fNewVel = VEL_MAX;
    if (fNewVel < VEL_MIN) fNewVel = VEL_MIN;
    particle.current_pos.x += fNewVel;
    particle.velocity.x = fNewVel;

    if (particle.current_pos.x > XMAX) particle.current_pos.x = XMAX;
    if (particle.current_pos.x < XMIN) particle.current_pos.x = XMIN;
    if (particle.current_pos.y > YMAX) particle.current_pos.y = YMAX;
    if (particle.current_pos.y < YMIN) particle.current_pos.y = YMIN;

    fitness(particle);
    //printParticle(particle);

    if (bRoot) {
      Particle tmpParticle;
      for (int iParticle = 1; iParticle < iSwarmSize; iParticle++) {
        MPI_Recv(&tmpParticle.current_fitness, 1, MPI_FLOAT, iParticle, MESSAGE_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&tmpParticle.current_pos.x,   1, MPI_FLOAT, iParticle, MESSAGE_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&tmpParticle.current_pos.y,   1, MPI_FLOAT, iParticle, MESSAGE_TAG, MPI_COMM_WORLD, &status);
      
        if (tmpParticle.current_fitness > pSwarmBest.best_fitness) {
          //iSwarmBest              = iParticle;
          pSwarmBest.best_fitness = tmpParticle.current_fitness; 
          pSwarmBest.best_pos.x   = tmpParticle.current_pos.x;
          pSwarmBest.best_pos.y   = tmpParticle.current_pos.y;
        }
      }
    } else {
      MPI_Send(&particle.current_fitness, 1, MPI_FLOAT, ROOT, MESSAGE_TAG, MPI_COMM_WORLD);
      MPI_Send(&particle.current_pos.x,   1, MPI_FLOAT, ROOT, MESSAGE_TAG, MPI_COMM_WORLD);
      MPI_Send(&particle.current_pos.y,   1, MPI_FLOAT, ROOT, MESSAGE_TAG, MPI_COMM_WORLD);
    } 
    
    MPI_Bcast(&pSwarmBest.best_fitness, 1, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&pSwarmBest.best_pos.x,   1, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&pSwarmBest.best_pos.y,   1, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    if (bRoot) {
      printf("SwarmBest: ");
      printParticle(pSwarmBest);
    }
  }

  if (bRoot) {
    printf("And the winner is: (%f, %f) -> %f\n",  pSwarmBest.best_pos.x, pSwarmBest.best_pos.y, pSwarmBest.best_fitness);
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
    return (rand() % (max+1))
}
//-------------------------------------------------
float fitness(struct Particle& f, bool bSetBest)
{
    float x = f.current_pos.x;
    float y = f.current_pos.y;

    f.current_fitness = -1*pow(x, 2);
    //f.current_fitness =  -1 * pow(x - 0.5, 3) + x;
    //f.current_fitness =  -1 * pow(x - 0.5, 3) + x - pow(y, 2) - (x * y);

    if (f.current_fitness > f.best_fitness || bSetBest == true) {
      f.best_fitness = f.current_fitness;
      f.best_pos.x   = x;
      f.best_pos.y   = y;
    }
    return f.current_fitness;
}
//-------------------------------------------------
void sendParticle(Particle *p, int destination)
{
    MPI_Send(&(p->current_fitnest), 1, MPI_FLOAT, destination, 100, MPI_COMM_WORLD);

    MPI_Send(&(p->current_pos.n), 1, MPI_INT, destination, 101, MPI_COMM_WORLD);
    MPI_Send(&(p->current_pos.m), 1, MPI_INT, destination, 102, MPI_COMM_WORLD);
    MPI_Send(&(p->current_pos.e), 1, MPI_FLOAT, destination, 103, MPI_COMM_WORLD);
    MPI_Send(&(p->current_pos.eb), 1, MPI_FLOAT, destination, 104, MPI_COMM_WORLD);

    MPI_Send(&(p->velocity.n), 1, MPI_INT, destination, 105, MPI_COMM_WORLD);
    MPI_Send(&(p->velocity.m), 1, MPI_INT, destination, 106, MPI_COMM_WORLD);
    MPI_Send(&(p->velocity.e), 1, MPI_FLOAT, destination, 107, MPI_COMM_WORLD);
    MPI_Send(&(p->velocity,eb), 1, MPI_FLOAT, destination, 108, MPI_COMM_WORLD);
}

void recvParticle(Particle *p, int source)
{
    MPI_Status status;
    MPI_Recv(&(p->current_fitnest), 1, MPI_FLOAT, source, 100,  MPI_COMM_WORLD, &status);

    MPI_Recv(&(p->current_pos.n), 1, MPI_INT, source, 101, MPI_COMM_WORLD, &status);
    MPI_Recv(&(p->current_pos.m), 1, MPI_INT, source, 102, MPI_COMM_WORLD, &status);
    MPI_Recv(&(p->current_pos.e), 1, MPI_FLOAT, source, 103, MPI_COMM_WORLD, &status);
    MPI_Recv(&(p->current_post.eb), 1, MPI_FLOAT, source, 104, MPI_COMM_WORLD, &status);

    MPI_Recv(&(p->velocity.n), 1, MPI_INT, source, 105, MPI_COMM_WORLD, &status);
    MPI_Recv(&(p->velocity.m), 1, MPI_INT, source, 106, MPI_COMM_WORLD, &status);
    MPI_Recv(&(p->velocity.e), 1, MPI_FLOAT, source, 107, MPI_COMM_WORLD, &status);
    MPI_Recv(&(p->velocity.eb), 1, MPI_FLOAT, source, 108, MPI_COMM_WORLD, &status);
}

void bcastParticle(Particle *p, int source)
{
    
}

//-------------------------------------------------
void printParticle(struct Particle p)
{
  printf("(%f @ %f) -> %f  best: (%f) -> %f\n", p.current_pos.x, p.velocity.x, p.current_fitness, p.best_pos.x, p.best_fitness);
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


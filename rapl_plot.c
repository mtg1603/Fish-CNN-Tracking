/** 
 * @author  Vince Weaver
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "papi.h"

#define MAX_EVENTS 128

char events[MAX_EVENTS][BUFSIZ];
char filenames[MAX_EVENTS][BUFSIZ];

FILE *fff[MAX_EVENTS];

static int num_events=0;

int main (int argc, char **argv)
{

    int retval,cid,rapl_cid=-1,numcmp;
    int EventSet = PAPI_NULL;
    long long values[MAX_EVENTS];
    int i,code,enum_retval;
    const PAPI_component_info_t *cmpinfo = NULL;
    long long start_time,before_time,after_time;
    double elapsed_time,total_time;
    char event_name[BUFSIZ];

	/* PAPI Initialization */
     retval = PAPI_library_init( PAPI_VER_CURRENT );
     if ( retval != PAPI_VER_CURRENT ) {
        fprintf(stderr,"PAPI_library_init failed\n");
	exit(1);
     }

     numcmp = PAPI_num_components();

     for(cid=0; cid<numcmp; cid++) {

	if ( (cmpinfo = PAPI_get_component_info(cid)) == NULL) {
	   fprintf(stderr,"PAPI_get_component_info failed\n");
	   exit(1);
	}

	if (strstr(cmpinfo->name,"rapl")) {
	   rapl_cid=cid;
	   printf("Found rapl component at cid %d\n", rapl_cid);

           if (cmpinfo->disabled) {
	     fprintf(stderr,"No rapl events found: %s\n",
		     cmpinfo->disabled_reason);
	     exit(1);
           }
	   break;
	}
     }

     /* Component not found */
     if (cid==numcmp) {
        fprintf(stderr,"No rapl component found\n");
        exit(1);
     }

     /* Find Events */
     code = PAPI_NATIVE_MASK;

     enum_retval = PAPI_enum_cmp_event( &code, PAPI_ENUM_FIRST, cid );

     while ( enum_retval == PAPI_OK ) {

       retval = PAPI_event_code_to_name( code, event_name );
       if ( retval != PAPI_OK ) {
	  printf("Error translating %#x\n",code);
	  exit(1);
       }

       printf("Found: %s\n",event_name);
       strncpy(events[num_events],event_name,BUFSIZ);
       sprintf(filenames[num_events],"results.%s",event_name);
       num_events++;

       if (num_events==MAX_EVENTS) {
	  printf("Too many events! %d\n",num_events);
	  exit(1);
       }

       enum_retval = PAPI_enum_cmp_event( &code, PAPI_ENUM_EVENTS, cid );
     }

     if (num_events==0) {
        printf("Error!  No RAPL events found!\n");
	exit(1);
     }

     /* Create EventSet */
     retval = PAPI_create_eventset( &EventSet );
     if (retval != PAPI_OK) {
        fprintf(stderr,"Error creating eventset!\n");
     }

     for(i=0;i<num_events;i++) {
        retval = PAPI_add_named_event( EventSet, events[i] );
        if (retval != PAPI_OK) {
	   fprintf(stderr,"Error adding event %s\n",events[i]);
	}
     }

     start_time=PAPI_get_real_nsec();

     while(1) {
        /* Start Counting */
        before_time=PAPI_get_real_nsec();
        retval = PAPI_start( EventSet);
        if (retval != PAPI_OK) {
           fprintf(stderr,"PAPI_start() failed\n");
	   exit(1);
        }

        usleep(33334);

        /* Stop Counting */
        after_time=PAPI_get_real_nsec();
        retval = PAPI_stop( EventSet, values);
        if (retval != PAPI_OK) {
           fprintf(stderr, "PAPI_start() failed\n");
        }

        total_time=((double)(after_time-start_time))/1.0e9;
        elapsed_time=((double)(after_time-before_time))/1.0e9;

        for(i=0;i<num_events;i++) {

	  printf("%.4f %.1f (* Average Power for %s *)\n",
		 total_time,
		 ((double)values[i]/1.0e9)/elapsed_time,
		 events[i]);
        }
     }

     return 0;
}

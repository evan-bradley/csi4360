    printf("========================================================================================\n");
    printf("\tMaximum Element Search in A[N][N], N=%d, %d tasks for omp_parallel\n", N, num_tasks);
    printf("----------------------------------------------------------------------------------------\n");
    printf("Performance:\t\t\tRuntime (ms) \t\tError (compared to base)\n");
    printf("----------------------------------------------------------------------------------------\n");
    printf("max_base:\t\t\t%4f \t\t%d\n",
           time * 1.0e3, max_base - max_base);
    printf("max_p:\t\t\t\t%4f \t\t%d\n",
           time_parallel * 1.0e3,  max_base - max_p);
    printf("max_p_for:\t\t\t%4f \t\t%d\n",
           time_parallel_for * 1.0e3, max_base - max_p_for);
    printf("max_p_for_reduc:\t\t%4f \t\t%d\n",
           time_parallel_for_reduction * 1.0e3, max_base - max_p_for_red);
    printf("hist_base:\t\t\t%4f \t\t%d\n",
           time_hist * 1.0e3, check_bins(bins, bins));
    printf("hist_p:\t\t\t\t%4f\t\t%d\n",
           time_hist_parallel * 1.0e3, check_bins(bins, bins_parallel));
    printf("hist_p_for:\t\t\t%4f \t\t%d\n",
           time_hist_parallel_for * 1.0e3, check_bins(bins, bins_parallel_for));
    printf("hist_p_for_reduc:\t\t%4f \t\t%d\n",
           time_hist_parallel_for_reduction * 1.0e3,
           check_bins(bins, bins_parallel_for_reduction));


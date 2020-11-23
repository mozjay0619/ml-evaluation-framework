import psutil


DEFAULT_LARGE_INSTANCE_WORKER_VCORES = 4
DEFAULT_SMALL_INSTANCE_WORKER_VCORES = 2

DASK_RESOURCE_PARAMETERS = [
    'local_client_n_workers', 
    'local_client_threads_per_worker', 
    'yarn_container_n_workers',
    'yarn_container_worker_vcores', 
    'yarn_container_worker_memory', 
    'n_worker_nodes']

INSTANCE_TYPES = {
    'm4.large': {'vCPU': 2, 'Mem': 8},
    'm4.xlarge': {'vCPU': 4, 'Mem': 16}, 
    'm4.2xlarge': {'vCPU': 8, 'Mem': 32},
    'm4.4xlarge': {'vCPU': 16, 'Mem': 64},
    'm4.10xlarge': {'vCPU': 40, 'Mem': 160},
    'm4.16xlarge': {'vCPU': 64, 'Mem': 256}, 
    
    'c4.large': {'vCPU': 2, 'Mem': 3.75},
    'c4.xlarge': {'vCPU': 4, 'Mem': 7.5},
    'c4.2xlarge': {'vCPU': 8, 'Mem': 15},
    'c4.4xlarge': {'vCPU': 16, 'Mem': 30},
    'c4.8xlarge': {'vCPU': 36, 'Mem': 60}, 
    
    'r4.large': {'vCPU': 2, 'Mem': 15.25},
    'r4.xlarge': {'vCPU': 4, 'Mem': 30.5},
    'r4.2xlarge': {'vCPU': 8, 'Mem': 61}, 
    'r4.4xlarge': {'vCPU': 16, 'Mem': 122},
    'r4.8xlarge': {'vCPU': 32, 'Mem': 244},
    'r4.16xlarge': {'vCPU': 64, 'Mem': 488}}

class DaskResourceConfigurer():

    def __init__(self):

        self.local_client_n_workers=None
        self.local_client_threads_per_worker=None
        self.yarn_client_n_workers=None
        self.yarn_client_worker_vcores=None
        self.yarn_client_worker_memory=None

        self.use_yarn_cluster = False

    def validate_dask_resource_configs(self, local_client_n_workers, local_client_threads_per_worker, 
        yarn_container_n_workers, yarn_container_worker_vcores, yarn_container_worker_memory,
        n_worker_nodes, use_yarn_cluster, use_ec2_instance, use_auto_config, instance_type, use_dashboard):

        local_client_resources_set = False
        yarn_client_resources_set = False

        # local resources were set manually by user
        if (local_client_n_workers is not None and 
            local_client_threads_per_worker is not None):
            local_client_resources_set = True

        # yarn resources were set manually by user
        # it 
        if (yarn_container_n_workers is not None and 
            yarn_container_worker_vcores is not None and
            yarn_container_worker_memory is not None and
            n_worker_nodes is not None):
            yarn_client_resources_set = True

        if local_client_resources_set:
            self.local_client_n_workers = local_client_n_workers
            self.local_client_threads_per_worker = local_client_threads_per_worker

            if yarn_client_resources_set:
                self.yarn_container_n_workers = yarn_container_n_workers
                self.yarn_container_worker_vcores = yarn_container_worker_vcores
                self.yarn_container_worker_memory = yarn_container_worker_memory
                self.n_worker_nodes = n_worker_nodes
            return

        if use_auto_config is None:
            print('\u2714 Set [ use_auto_config ] to True in order to automatically configure Dask resources.\n')

        if use_yarn_cluster is None:
            print('\u2714 Set [ use_yarn_cluster ] to True in order to leverage Yarn cluster.\n')

        if use_ec2_instance is None:
            print('\u2714 Set [ use_ec2_instance ] to True in order to leverage EC2 instance.\n')

        if use_auto_config is None:
            print('\u27AA You can also manually configure resources by providing arguments for the following '
                      'parameters:\n\n\u25BA {}'.format('  '.join(DASK_RESOURCE_PARAMETERS[0:4])))
            print('\n  ' + '  '.join(DASK_RESOURCE_PARAMETERS[4:]))

        # optional arguments
        optional_args = []

        if (use_auto_config is None) or (use_yarn_cluster is None):
            optional_args.append('instance_type')

        self.use_dashboard = use_dashboard
        optional_args.append('use_dashboard')

        if len(optional_args)>0:
            print('\nOptional argument(s):\n\n\u25BA {}'.format('  '.join(optional_args)))


#         if (use_auto_config is None) or (use_yarn_cluster is None):
#             print('\nOptional argument(s):\n\n\u25BA {}'.format('instance_type'))



# if len(self.required_args)>0 or len(self.ordered_CV_required_args)>0:
#             if len(self.optional_args)>0:
#                 print('\nOptional argument(s):\n\n\u25BA {}'.format('  '.join(self.optional_args)))
#                 print()



        if use_auto_config:

            if use_yarn_cluster or use_ec2_instance:

                if (instance_type is None or n_worker_nodes is None):

                    if use_yarn_cluster:

                        print('\n\u2714 In order to auto config yarn cluster, please provide the [ instance_type ] '
                              'and [ n_worker_nodes ].')
                        print('EX: instance_type="m4.2xlarge", n_worker_nodes=3 (excluding the master node)')

                    if (use_ec2_instance):

                        print('\n\u2714 In order to auto config ec2 instance, please provide the [ instance_type ] (EX: instance_type="m4.2xlarge")')
                        # print('')

                    print('\nAvailable [ instance_type ] options: ')
                    print('\n  {}'.format('  '.join(list(INSTANCE_TYPES.keys())[0:6])))
                    print('\n  ' + '  '.join(list(INSTANCE_TYPES.keys())[6:11]))
                    print('\n  ' + '  '.join(list(INSTANCE_TYPES.keys())[11:]))
                    return



                if use_yarn_cluster:

                    self.use_yarn_cluster = True

                else:
                    num_physical_cores = int(INSTANCE_TYPES[instance_type]['vCPU']/2)
                    num_virtual_cores = int(INSTANCE_TYPES[instance_type]['vCPU'])
                    available_memory = int(INSTANCE_TYPES[instance_type]['Mem'] - 4)
                    # 2 GB claimed by client + 2 GB claimed by scheduler in a node

                    large_instance = num_physical_cores>=8

                    if large_instance:

                        local_offset = 4
                        self.local_client_threads_per_worker = DEFAULT_LARGE_INSTANCE_WORKER_VCORES
                        self.local_client_n_workers = int((num_virtual_cores - 
                                                      local_offset)/self.local_client_threads_per_worker)
                        
                        if use_yarn_cluster:

                            yarn_offset = 2
                            self.yarn_container_worker_vcores = DEFAULT_LARGE_INSTANCE_WORKER_VCORES
                            self.yarn_container_n_workers = int((num_virtual_cores - 
                                                                 yarn_offset)/self.yarn_container_worker_vcores)
                            self.yarn_container_worker_memory = str(int((available_memory - 
                                                                    1.5)/self.yarn_container_n_workers)) + ' GB'
                            self.yarn_container_worker_memory = str(int((available_memory - 
                                                                    1.5)/self.yarn_container_n_workers)) + ' GB'

                    else:

                        local_offset = 2
                        self.local_client_threads_per_worker = DEFAULT_SMALL_INSTANCE_WORKER_VCORES
                        self.local_client_n_workers = int(max(1, num_virtual_cores - 
                                                         local_offset)/self.local_client_threads_per_worker)

                        if use_yarn_cluster:

                            yarn_offset = 2
                            self.yarn_container_worker_vcores = DEFAULT_SMALL_INSTANCE_WORKER_VCORES
                            self.yarn_container_n_workers = int(max(1, num_virtual_cores - 
                                                               yarn_offset)/self.yarn_container_worker_vcores)
                            self.yarn_container_worker_memory = str(int((available_memory - 
                                                                    1.5)/self.yarn_container_n_workers)) + ' GB'

                if use_yarn_cluster:

                    self.n_worker_nodes = n_worker_nodes
                
                print('[ aws instance configurations ]')
                print('instance vcores: {}'.format(INSTANCE_TYPES[instance_type]['vCPU']))
                print('instance memory: {} GB'.format(INSTANCE_TYPES[instance_type]['Mem']))
                print('n_worker_nodes: {}'.format(self.n_worker_nodes))
                print()
                print('[ dask configurations ]')
                print('local_client_n_workers: {}'.format(self.local_client_n_workers))
                print('local_client_threads_per_worker: {}'.format(self.local_client_threads_per_worker))

                if use_yarn_cluster:
                    print('yarn_container_n_workers: {}'.format(self.yarn_container_n_workers))
                    print('yarn_container_worker_vcores: {}'.format(self.yarn_container_worker_vcores))
                    print('yarn_container_worker_memory: {}'.format(self.yarn_container_worker_memory))

            else:

                self.use_yarn_cluster = False
                
                self.local_client_n_workers = psutil.cpu_count(logical=False)
                self.local_client_threads_per_worker = int(psutil.cpu_count(logical=True)/self.local_client_n_workers)
                
                print('[ dask configurations ]')
                print('local_client_n_workers: {}'.format(self.local_client_n_workers))
                print('local_client_threads_per_worker: {}'.format(self.local_client_threads_per_worker))



                
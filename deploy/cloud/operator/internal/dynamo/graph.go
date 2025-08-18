/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dynamo

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"regexp"
	"sort"
	"strings"

	istioNetworking "istio.io/api/networking/v1beta1"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"

	grovev1alpha1 "github.com/NVIDIA/grove/operator/api/core/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/dynamo/common"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/controller_common"
	"github.com/imdario/mergo"
	networkingv1beta1 "istio.io/client-go/pkg/apis/networking/v1beta1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
)

// ServiceConfig represents the YAML configuration structure for a service
type DynamoConfig struct {
	Enabled       bool   `yaml:"enabled"`
	Namespace     string `yaml:"namespace"`
	Name          string `yaml:"name"`
	ComponentType string `yaml:"component_type,omitempty"`
}

type Traffic struct {
	Timeout int `yaml:"timeout"`
}

type Autoscaling struct {
	MinReplicas int `yaml:"min_replicas"`
	MaxReplicas int `yaml:"max_replicas"`
}

type Config struct {
	Dynamo       *DynamoConfig        `yaml:"dynamo,omitempty"`
	Resources    *Resources           `yaml:"resources,omitempty"`
	Traffic      *Traffic             `yaml:"traffic,omitempty"`
	Autoscaling  *Autoscaling         `yaml:"autoscaling,omitempty"`
	HttpExposed  bool                 `yaml:"http_exposed,omitempty"`
	ApiEndpoints []string             `yaml:"api_endpoints,omitempty"`
	Workers      *int32               `yaml:"workers,omitempty"`
	TotalGpus    *int32               `yaml:"total_gpus,omitempty"`
	ExtraPodSpec *common.ExtraPodSpec `yaml:"extraPodSpec,omitempty"`
}

type ServiceConfig struct {
	Name         string              `yaml:"name"`
	Dependencies []map[string]string `yaml:"dependencies,omitempty"`
	Config       Config              `yaml:"config"`
}

type Resources struct {
	CPU    *string           `yaml:"cpu,omitempty" json:"cpu,omitempty"`
	Memory *string           `yaml:"memory,omitempty" json:"memory,omitempty"`
	GPU    *string           `yaml:"gpu,omitempty" json:"gpu,omitempty"`
	Custom map[string]string `yaml:"custom,omitempty" json:"custom,omitempty"`
}

type DynDeploymentConfig = map[string]*DynDeploymentServiceConfig

// ServiceConfig represents the configuration for a specific service
type DynDeploymentServiceConfig struct {
	ServiceArgs *ServiceArgs `json:"ServiceArgs,omitempty"`
}

// ServiceArgs represents the arguments that can be passed to any service
type ServiceArgs struct {
	Workers   *int32     `json:"workers,omitempty"`
	Resources *Resources `json:"resources,omitempty"`
}

func (s ServiceConfig) GetNamespace() *string {
	if s.Config.Dynamo == nil || s.Config.Dynamo.Namespace == "" {
		return nil
	}
	return &s.Config.Dynamo.Namespace
}

func GetDefaultDynamoNamespace(dynamoDeployment *v1alpha1.DynamoGraphDeployment) string {
	return fmt.Sprintf("dynamo-%s", dynamoDeployment.Name)
}

func ParseDynDeploymentConfig(ctx context.Context, jsonContent []byte) (DynDeploymentConfig, error) {
	var config DynDeploymentConfig
	err := json.Unmarshal(jsonContent, &config)
	return config, err
}

// GenerateDynamoComponentsDeployments generates a map of DynamoComponentDeployments from a DynamoGraphConfig
func GenerateDynamoComponentsDeployments(ctx context.Context, parentDynamoGraphDeployment *v1alpha1.DynamoGraphDeployment, defaultIngressSpec *v1alpha1.IngressSpec) (map[string]*v1alpha1.DynamoComponentDeployment, error) {
	deployments := make(map[string]*v1alpha1.DynamoComponentDeployment)
	graphDynamoNamespace, err := getDynamoNamespace(parentDynamoGraphDeployment)
	if err != nil {
		return nil, fmt.Errorf("failed to get the graph dynamo namespace: %w", err)
	}
	for componentName, component := range parentDynamoGraphDeployment.Spec.Services {
		deployment := &v1alpha1.DynamoComponentDeployment{}
		deployment.Spec.DynamoComponentDeploymentSharedSpec = component.DynamoComponentDeploymentSharedSpec
		deployment.Name = GetDynamoComponentName(parentDynamoGraphDeployment, componentName)
		deployment.Spec.BackendFramework = parentDynamoGraphDeployment.Spec.BackendFramework
		deployment.Namespace = parentDynamoGraphDeployment.Namespace
		deployment.Spec.ServiceName = componentName
		deployment.Spec.DynamoNamespace = &graphDynamoNamespace
		labels := make(map[string]string)
		// add the labels in the spec in order to label all sub-resources
		deployment.Spec.Labels = labels
		// and add the labels to the deployment itself
		deployment.Labels = labels
		labels[commonconsts.KubeLabelDynamoComponent] = componentName
		labels[commonconsts.KubeLabelDynamoNamespace] = graphDynamoNamespace

		// Propagate metrics annotation from parent deployment if present
		if parentDynamoGraphDeployment.Annotations != nil {
			if val, exists := parentDynamoGraphDeployment.Annotations[commonconsts.KubeAnnotationEnableMetrics]; exists {
				if deployment.Spec.Annotations == nil {
					deployment.Spec.Annotations = make(map[string]string)
				}
				deployment.Spec.Annotations[commonconsts.KubeAnnotationEnableMetrics] = val
			}
		}

		if component.ComponentType == commonconsts.ComponentTypePlanner {
			// ensure that the extraPodSpec is not nil
			if deployment.Spec.ExtraPodSpec == nil {
				deployment.Spec.ExtraPodSpec = &common.ExtraPodSpec{}
			}
			// ensure that the embedded PodSpec struct is not nil
			if deployment.Spec.ExtraPodSpec.PodSpec == nil {
				deployment.Spec.ExtraPodSpec.PodSpec = &corev1.PodSpec{}
			}
			// finally set the service account name
			deployment.Spec.ExtraPodSpec.PodSpec.ServiceAccountName = commonconsts.PlannerServiceAccountName
		}
		if deployment.IsFrontendComponent() && defaultIngressSpec != nil && deployment.Spec.Ingress == nil {
			deployment.Spec.Ingress = defaultIngressSpec
		}
		// merge the envs from the parent deployment with the envs from the service
		if len(parentDynamoGraphDeployment.Spec.Envs) > 0 {
			deployment.Spec.Envs = MergeEnvs(parentDynamoGraphDeployment.Spec.Envs, deployment.Spec.Envs)
		}
		err := updateDynDeploymentConfig(deployment, commonconsts.DynamoServicePort)
		if err != nil {
			return nil, err
		}
		err = overrideWithDynDeploymentConfig(ctx, deployment)
		if err != nil {
			return nil, err
		}
		// we only override the replicas if it is not set in the CRD.
		// replicas, if set in the CRD must always be the source of truth.
		if component.Replicas != nil {
			deployment.Spec.Replicas = component.Replicas
		}
		deployments[componentName] = deployment
	}
	return deployments, nil
}

func getDynamoNamespace(parentDynamoGraphDeployment *v1alpha1.DynamoGraphDeployment) (string, error) {
	graphDynamoNamespace := ""
	for componentName, component := range parentDynamoGraphDeployment.Spec.Services {
		dynamoNamespace := ""
		if component.DynamoNamespace != nil && *component.DynamoNamespace != "" {
			dynamoNamespace = *component.DynamoNamespace
		}
		if graphDynamoNamespace != "" && graphDynamoNamespace != dynamoNamespace {
			return "", fmt.Errorf("namespace mismatch for component %s: graph uses namespace %s but component specifies %s", componentName, graphDynamoNamespace, dynamoNamespace)
		}
		graphDynamoNamespace = dynamoNamespace
	}
	if graphDynamoNamespace == "" {
		graphDynamoNamespace = GetDefaultDynamoNamespace(parentDynamoGraphDeployment)
	}
	return graphDynamoNamespace, nil
}

// updateDynDeploymentConfig updates the runtime config object for the given dynamoDeploymentComponent
// It updates the port for the given service (if it is the main component)
func updateDynDeploymentConfig(dynamoDeploymentComponent *v1alpha1.DynamoComponentDeployment, newPort int) error {
	if dynamoDeploymentComponent.IsFrontendComponent() {
		dynamoDeploymentConfig := dynamoDeploymentComponent.GetDynamoDeploymentConfig()
		if dynamoDeploymentConfig != nil {
			var config map[string]any
			if err := json.Unmarshal(dynamoDeploymentConfig, &config); err != nil {
				return fmt.Errorf("failed to unmarshal %v: %w", commonconsts.DynamoDeploymentConfigEnvVar, err)
			}
			// Safely navigate and update the config
			if serviceConfig, ok := config[dynamoDeploymentComponent.Spec.ServiceName].(map[string]any); ok {
				serviceConfig["port"] = newPort
			}
			// Marshal back to JSON string
			updated, err := json.Marshal(config)
			if err != nil {
				return fmt.Errorf("failed to marshal updated config: %w", err)
			}
			dynamoDeploymentComponent.SetDynamoDeploymentConfig(updated)
		}
	}
	return nil
}

func overrideWithDynDeploymentConfig(ctx context.Context, dynamoDeploymentComponent *v1alpha1.DynamoComponentDeployment) error {
	dynamoDeploymentConfig := dynamoDeploymentComponent.GetDynamoDeploymentConfig()
	if dynamoDeploymentConfig == nil {
		return nil
	}
	dynDeploymentConfig, err := ParseDynDeploymentConfig(ctx, dynamoDeploymentConfig)
	if err != nil {
		return fmt.Errorf("failed to parse %v: %w", commonconsts.DynamoDeploymentConfigEnvVar, err)
	}
	componentDynConfig := dynDeploymentConfig[dynamoDeploymentComponent.Spec.ServiceName]
	if componentDynConfig != nil {
		if componentDynConfig.ServiceArgs != nil && componentDynConfig.ServiceArgs.Workers != nil {
			dynamoDeploymentComponent.Spec.Replicas = componentDynConfig.ServiceArgs.Workers
		}
		if componentDynConfig.ServiceArgs != nil && componentDynConfig.ServiceArgs.Resources != nil {
			requests := &common.ResourceItem{}
			limits := &common.ResourceItem{}
			if dynamoDeploymentComponent.Spec.Resources == nil {
				dynamoDeploymentComponent.Spec.Resources = &common.Resources{
					Requests: requests,
					Limits:   limits,
				}
			} else {
				if dynamoDeploymentComponent.Spec.Resources.Requests != nil {
					requests = dynamoDeploymentComponent.Spec.Resources.Requests
				} else {
					dynamoDeploymentComponent.Spec.Resources.Requests = requests
				}
				if dynamoDeploymentComponent.Spec.Resources.Limits != nil {
					limits = dynamoDeploymentComponent.Spec.Resources.Limits
				} else {
					dynamoDeploymentComponent.Spec.Resources.Limits = limits
				}
			}
			if componentDynConfig.ServiceArgs.Resources.GPU != nil {
				requests.GPU = *componentDynConfig.ServiceArgs.Resources.GPU
				limits.GPU = *componentDynConfig.ServiceArgs.Resources.GPU
			}
			if componentDynConfig.ServiceArgs.Resources.CPU != nil {
				requests.CPU = *componentDynConfig.ServiceArgs.Resources.CPU
				limits.CPU = *componentDynConfig.ServiceArgs.Resources.CPU
			}
			if componentDynConfig.ServiceArgs.Resources.Memory != nil {
				requests.Memory = *componentDynConfig.ServiceArgs.Resources.Memory
				limits.Memory = *componentDynConfig.ServiceArgs.Resources.Memory
			}
			if componentDynConfig.ServiceArgs.Resources.Custom != nil {
				requests.Custom = componentDynConfig.ServiceArgs.Resources.Custom
				limits.Custom = componentDynConfig.ServiceArgs.Resources.Custom
			}
		}
	}
	return nil
}

func MergeEnvs(common, specific []corev1.EnvVar) []corev1.EnvVar {
	envMap := make(map[string]corev1.EnvVar)

	// Add all common environment variables.
	for _, env := range common {
		envMap[env.Name] = env
	}

	// Override or add with service-specific environment variables.
	for _, env := range specific {
		envMap[env.Name] = env
	}

	// Convert the map back to a slice.
	merged := make([]corev1.EnvVar, 0, len(envMap))
	for _, env := range envMap {
		merged = append(merged, env)
	}
	sort.Slice(merged, func(i, j int) bool {
		return merged[i].Name < merged[j].Name
	})
	return merged
}

func GetDynamoComponentName(dynamoDeployment *v1alpha1.DynamoGraphDeployment, component string) string {
	return fmt.Sprintf("%s-%s", dynamoDeployment.Name, strings.ToLower(component))
}

type SecretsRetriever interface {
	GetSecrets(namespace, registry string) ([]string, error)
}

// applyCliqueStartupDependencies configures StartsAfter dependencies for cliques in a PodGangSet
// based on the backend framework and multinode deployment patterns.
//
// Rules:
// - For VLLM and SGLang: worker cliques start after leader clique
// - For TRTLLM: leader clique starts after worker cliques
// - Only applies to multinode deployments (numberOfNodes > 1)
// - Sets the PodGangSet StartupType to Explicit if any dependencies are configured
func applyCliqueStartupDependencies(
	gangSet *grovev1alpha1.PodGangSet,
	roles []ServiceRole,
	backendFramework BackendFramework,
	numberOfNodes int32,
) {
	// deactivated for now.
	// TODO: reactivate this when we have a better way to handle the readiness probe for the leader.
	deactivated := true

	if deactivated || numberOfNodes <= 1 {
		return // No dependencies for single-node deployments
	}

	// Build maps of leader and worker clique names
	var leaderCliqueName string
	var workerCliqueNames []string

	for _, r := range roles {
		cliqueName := strings.ToLower(r.Name)
		switch r.Role {
		case RoleLeader:
			leaderCliqueName = cliqueName
		case RoleWorker:
			workerCliqueNames = append(workerCliqueNames, cliqueName)
		}
	}

	// Apply dependencies to cliques
	hasDependencies := false
	for _, clique := range gangSet.Spec.Template.Cliques {
		// Find the corresponding role for this clique
		var cliqueRole Role
		for _, r := range roles {
			if strings.ToLower(r.Name) == clique.Name {
				cliqueRole = r.Role
				break
			}
		}

		// Determine dependencies for this clique
		startsAfter := getCliqueStartupDependencies(cliqueRole, backendFramework, leaderCliqueName, workerCliqueNames)
		if len(startsAfter) > 0 {
			clique.Spec.StartsAfter = startsAfter
			hasDependencies = true
		}
	}

	// Set explicit startup type if we have any dependencies
	if hasDependencies {
		explicitStartupType := grovev1alpha1.CliqueStartupTypeExplicit
		gangSet.Spec.Template.StartupType = &explicitStartupType
	}
}

// getCliqueStartupDependencies determines the StartsAfter dependencies for a clique
// based on its role, backend framework, and available leader/worker clique names.
//
// Rules:
// - For VLLM and SGLang: worker cliques start after leader clique
// - For TRTLLM: leader clique starts after worker cliques
// - For other backends or single-node deployments: no dependencies
func getCliqueStartupDependencies(
	role Role,
	backendFramework BackendFramework,
	leaderCliqueName string,
	workerCliqueNames []string,
) []string {
	switch backendFramework {
	case BackendFrameworkVLLM, BackendFrameworkSGLang:
		// For vllm and sglang: worker cliques start after leader clique
		if role == RoleWorker && leaderCliqueName != "" {
			return []string{leaderCliqueName}
		}
	case BackendFrameworkTRTLLM:
		// For trtllm: leader clique starts after worker cliques
		if role == RoleLeader && len(workerCliqueNames) > 0 {
			return workerCliqueNames
		}
	}

	// No dependencies for other cases
	return nil
}

func GenerateComponentService(ctx context.Context, componentName, componentNamespace string) (*corev1.Service, error) {
	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      componentName,
			Namespace: componentNamespace,
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{
				commonconsts.KubeLabelDynamoSelector: componentName,
			},
			Ports: []corev1.ServicePort{
				{
					Name:       commonconsts.DynamoServicePortName,
					Port:       commonconsts.DynamoServicePort,
					TargetPort: intstr.FromString(commonconsts.DynamoContainerPortName),
					Protocol:   corev1.ProtocolTCP,
				},
			},
		},
	}
	return service, nil
}

func GenerateComponentIngress(ctx context.Context, componentName, componentNamespace string, ingressSpec v1alpha1.IngressSpec) *networkingv1.Ingress {
	resourceName := componentName
	ingress := &networkingv1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:      resourceName,
			Namespace: componentNamespace,
		},
	}
	host := getIngressHost(ingressSpec)
	ingress.Spec = networkingv1.IngressSpec{
		IngressClassName: ingressSpec.IngressControllerClassName,
		Rules: []networkingv1.IngressRule{
			{
				Host: host,
				IngressRuleValue: networkingv1.IngressRuleValue{
					HTTP: &networkingv1.HTTPIngressRuleValue{
						Paths: []networkingv1.HTTPIngressPath{
							{
								Path:     "/",
								PathType: &[]networkingv1.PathType{networkingv1.PathTypePrefix}[0],
								Backend: networkingv1.IngressBackend{
									Service: &networkingv1.IngressServiceBackend{
										Name: resourceName,
										Port: networkingv1.ServiceBackendPort{
											Number: commonconsts.DynamoServicePort,
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	if ingressSpec.TLS != nil {
		ingress.Spec.TLS = []networkingv1.IngressTLS{
			{
				Hosts:      []string{host},
				SecretName: ingressSpec.TLS.SecretName,
			},
		}
	}
	return ingress
}

func getIngressHost(ingressSpec v1alpha1.IngressSpec) string {
	host := ingressSpec.Host
	if ingressSpec.HostPrefix != nil {
		host = *ingressSpec.HostPrefix + host
	}
	ingressSuffix := commonconsts.DefaultIngressSuffix
	if ingressSpec.HostSuffix != nil {
		ingressSuffix = *ingressSpec.HostSuffix
	}
	return fmt.Sprintf("%s.%s", host, ingressSuffix)
}

func GenerateComponentVirtualService(ctx context.Context, componentName, componentNamespace string, ingressSpec v1alpha1.IngressSpec) *networkingv1beta1.VirtualService {
	vs := &networkingv1beta1.VirtualService{
		ObjectMeta: metav1.ObjectMeta{
			Name:      componentName,
			Namespace: componentNamespace,
		},
	}
	if ingressSpec.IsVirtualServiceEnabled() {
		vs.Spec = istioNetworking.VirtualService{
			Hosts: []string{
				getIngressHost(ingressSpec),
			},
			Gateways: []string{*ingressSpec.VirtualServiceGateway},
			Http: []*istioNetworking.HTTPRoute{
				{
					Match: []*istioNetworking.HTTPMatchRequest{
						{
							Uri: &istioNetworking.StringMatch{
								MatchType: &istioNetworking.StringMatch_Prefix{Prefix: "/"},
							},
						},
					},
					Route: []*istioNetworking.HTTPRouteDestination{
						{
							Destination: &istioNetworking.Destination{
								Host: componentName,
								Port: &istioNetworking.PortSelector{
									Number: commonconsts.DynamoServicePort,
								},
							},
						},
					},
				},
			},
		}
	}
	return vs
}

func GenerateDefaultIngressSpec(dynamoDeployment *v1alpha1.DynamoGraphDeployment, ingressConfig controller_common.IngressConfig) v1alpha1.IngressSpec {
	res := v1alpha1.IngressSpec{
		Enabled:           ingressConfig.VirtualServiceGateway != "" || ingressConfig.IngressControllerClassName != "",
		Host:              dynamoDeployment.Name,
		UseVirtualService: ingressConfig.VirtualServiceGateway != "",
	}
	if ingressConfig.IngressControllerClassName != "" {
		res.IngressControllerClassName = &ingressConfig.IngressControllerClassName
	}
	if ingressConfig.IngressControllerTLSSecret != "" {
		res.TLS = &v1alpha1.IngressTLSSpec{
			SecretName: ingressConfig.IngressControllerTLSSecret,
		}
	}
	if ingressConfig.IngressHostSuffix != "" {
		res.HostSuffix = &ingressConfig.IngressHostSuffix
	}
	if ingressConfig.VirtualServiceGateway != "" {
		res.VirtualServiceGateway = &ingressConfig.VirtualServiceGateway
	}
	return res
}

// Helper: mergeContainerCommand returns userCmd if specified, else defaultCmd
func mergeContainerCommand(defaultCmd, userCmd []string) []string {
	if len(userCmd) > 0 {
		return userCmd
	}
	return defaultCmd
}

// Define Role enum for leader/worker/main
// Use this type everywhere instead of string for role

type Role string

const (
	RoleLeader Role = "leader"
	RoleWorker Role = "worker"
	RoleMain   Role = "main"
)

// Update ServiceRole struct for expandRolesForService

type ServiceRole struct {
	Name     string
	Role     Role
	Replicas int32
}

// Update expandRolesForService to use Role
func expandRolesForService(serviceName string, serviceReplicas *int32, numberOfNodes int32) []ServiceRole {
	var roles []ServiceRole
	if numberOfNodes > 1 {
		roles = append(roles, ServiceRole{Name: serviceName + "-" + commonconsts.GroveRoleSuffixLeader, Role: RoleLeader, Replicas: 1})
		roles = append(roles, ServiceRole{Name: serviceName + "-" + commonconsts.GroveRoleSuffixWorker, Role: RoleWorker, Replicas: numberOfNodes - 1})
	} else {
		roles = append(roles, ServiceRole{Name: serviceName, Role: RoleMain, Replicas: *serviceReplicas})
	}
	return roles
}

// Define BackendFramework enum for sglang, vllm, trtllm

type BackendFramework string

const (
	BackendFrameworkSGLang BackendFramework = "sglang"
	BackendFrameworkVLLM   BackendFramework = "vllm"
	BackendFrameworkTRTLLM BackendFramework = "trtllm"
)

// Backend interface for modular backend logic
// Each backend (SGLang, VLLM, etc.) implements this interface
type Backend interface {
	UpdateContainer(container *corev1.Container, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, serviceName string, multinodeDeployer MultinodeDeployer)
	UpdatePodSpec(podSpec *corev1.PodSpec, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, serviceName string)
}

// NoopBackend does no processing - used for non-worker components like frontend, planner, router
type NoopBackend struct{}

func (b *NoopBackend) UpdateContainer(container *corev1.Container, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, serviceName string, multinodeDeployer MultinodeDeployer) {
	// No-op: frontend, planner, router, etc. don't need backend-specific processing
}

func (b *NoopBackend) UpdatePodSpec(podSpec *corev1.PodSpec, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, serviceName string) {
	// No-op: frontend, planner, router, etc. don't need backend-specific processing
}

type MultinodeDeployer interface {
	GetLeaderHostname(serviceName string) string
	GetHostNames(serviceName string, numberOfNodes int32) []string
	GetNodeRank() string
}

// BackendFactory creates backend instances based on the framework type
func BackendFactory(backendFramework BackendFramework) Backend {
	switch backendFramework {
	case BackendFrameworkSGLang:
		return &SGLangBackend{}
	case BackendFrameworkVLLM:
		return &VLLMBackend{}
	case BackendFrameworkTRTLLM:
		return &TRTLLMBackend{}
	case BackendFrameworkNoop:
		return &NoopBackend{}
	default:
		return nil
	}
}

func MultinodeDeployerFactory(multinodeDeploymentType commonconsts.MultinodeDeploymentType) MultinodeDeployer {
	switch multinodeDeploymentType {
	case commonconsts.MultinodeDeploymentTypeGrove:
		return &GroveMultinodeDeployer{}
	case commonconsts.MultinodeDeploymentTypeLWS:
		return &LWSMultinodeDeployer{}
	default:
		return nil
	}
}

// isWorkerComponent checks if a component is a worker that needs backend framework detection
func isWorkerComponent(componentType string) bool {
	return componentType == commonconsts.ComponentTypeWorker
}

// addStandardEnvVars adds the standard environment variables that are common to both Grove and Controller
func addStandardEnvVars(container *corev1.Container, controllerConfig controller_common.Config) {
	if controllerConfig.NatsAddress != "" {
		container.Env = append(container.Env, corev1.EnvVar{
			Name:  "NATS_SERVER",
			Value: controllerConfig.NatsAddress,
		})
	}

	if controllerConfig.EtcdAddress != "" {
		container.Env = append(container.Env, corev1.EnvVar{
			Name:  "ETCD_ENDPOINTS",
			Value: controllerConfig.EtcdAddress,
		})
	}
}

// GenerateBasePodSpec creates a basic PodSpec with common logic shared between controller and grove
// Includes standard environment variables (DYNAMO_PORT, NATS_SERVER, ETCD_ENDPOINTS)
// Deployment-specific environment merging should be handled by the caller
func GenerateBasePodSpec(
	component *v1alpha1.DynamoComponentDeploymentOverridesSpec,
	backendFramework BackendFramework,
	secretsRetriever SecretsRetriever,
	parentGraphDeploymentName string,
	namespace string,
	role Role,
	numberOfNodes int32,
	controllerConfig controller_common.Config,
	multinodeDeploymentType commonconsts.MultinodeDeploymentType,
	serviceName string,
) (corev1.PodSpec, error) {
	// Start with base container generated per component type
	componentContext := generateComponentContext(component, parentGraphDeploymentName, namespace, numberOfNodes)
	componentDefaults := ComponentDefaultsFactory(component.ComponentType)
	container, err := componentDefaults.GetBaseContainer(componentContext)
	if err != nil {
		return corev1.PodSpec{}, fmt.Errorf("failed to get base container: %w", err)
	}

	if component.ExtraPodSpec != nil && component.ExtraPodSpec.MainContainer != nil {
		main := component.ExtraPodSpec.MainContainer.DeepCopy()
		if main != nil {
			// merge the extraPodSpec from the parent deployment with the extraPodSpec from the service
			err = mergo.Merge(&container, *main, mergo.WithOverride)
			if err != nil {
				return corev1.PodSpec{}, fmt.Errorf("failed to merge extraPodSpec: %w", err)
			}

			// main container fields that require special handling
			container.Env = MergeEnvs(component.Envs, container.Env)
			// Note: startup probe does not have its own top level field so it must be passed in extraPodSpec.MainContainer
			// We want to overwrite entirely if provided rather than merge
			if main.StartupProbe != nil {
				container.StartupProbe = main.StartupProbe
			}
		}
	}

	// Merge probes entirely if they are passed (no partial merge)
	if component.LivenessProbe != nil {
		container.LivenessProbe = component.LivenessProbe.DeepCopy()
	}
	if component.ReadinessProbe != nil {
		container.ReadinessProbe = component.ReadinessProbe.DeepCopy()
	}

	overrideResources, err := controller_common.GetResourcesConfig(component.Resources)
	if err != nil {
		return corev1.PodSpec{}, fmt.Errorf("failed to get resources config: %w", err)
	}
	// Requests
	if overrideResources != nil && len(overrideResources.Requests) > 0 {
		if container.Resources.Requests == nil {
			container.Resources.Requests = corev1.ResourceList{}
		}
		maps.Copy(container.Resources.Requests, overrideResources.Requests)
	}

	// Limits
	if overrideResources != nil && len(overrideResources.Limits) > 0 {
		if container.Resources.Limits == nil {
			container.Resources.Limits = corev1.ResourceList{}
		}
		maps.Copy(container.Resources.Limits, overrideResources.Limits)
	}

	imagePullSecrets := []corev1.LocalObjectReference{}
	if secretsRetriever != nil && component.ExtraPodSpec != nil && component.ExtraPodSpec.MainContainer != nil && component.ExtraPodSpec.MainContainer.Image != "" {
		secretsName, err := secretsRetriever.GetSecrets(namespace, component.ExtraPodSpec.MainContainer.Image)
		if err == nil {
			for _, secretName := range secretsName {
				imagePullSecrets = append(imagePullSecrets, corev1.LocalObjectReference{Name: secretName})
			}
		}
	}
	if component.EnvFromSecret != nil {
		container.EnvFrom = append(container.EnvFrom, corev1.EnvFromSource{
			SecretRef: &corev1.SecretEnvSource{
				LocalObjectReference: corev1.LocalObjectReference{Name: *component.EnvFromSecret},
			},
		})
	}

	addStandardEnvVars(&container, controllerConfig)

	var volumes []corev1.Volume
	if component.PVC != nil {
		volumes = append(volumes, corev1.Volume{
			Name: *component.PVC.Name,
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: *component.PVC.Name,
				},
			},
		})
		container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{
			Name:      *component.PVC.Name,
			MountPath: *component.PVC.MountPoint,
		})
	}
	shmVolume, shmVolumeMount := generateSharedMemoryVolumeAndMount(&container.Resources)
	volumes = append(volumes, shmVolume)
	container.VolumeMounts = append(container.VolumeMounts, shmVolumeMount)

	// Apply backend-specific container modifications
	multinodeDeployer := MultinodeDeployerFactory(multinodeDeploymentType)
	if multinodeDeployer == nil {
		return corev1.PodSpec{}, fmt.Errorf("unsupported multinode deployment type: %s", multinodeDeploymentType)
	}
	backend := BackendFactory(backendFramework)
	if backend == nil {
		return corev1.PodSpec{}, fmt.Errorf("unsupported backend framework: %s", backendFramework)
	}
	backend.UpdateContainer(&container, numberOfNodes, role, component, serviceName, multinodeDeployer)

	// get base podspec from component
	podSpec, err := componentDefaults.GetBasePodSpec(componentContext)
	if err != nil {
		return corev1.PodSpec{}, fmt.Errorf("failed to get base podspec: %w", err)
	}

	if component.ExtraPodSpec != nil && component.ExtraPodSpec.PodSpec != nil {
		// merge extraPodSpec PodSpec with base podspec
		err := mergo.Merge(&podSpec, component.ExtraPodSpec.PodSpec.DeepCopy(), mergo.WithOverride)
		if err != nil {
			return corev1.PodSpec{}, fmt.Errorf("failed to merge extraPodSpec: %w", err)
		}
	}
	podSpec.Containers = append(podSpec.Containers, container)
	podSpec.Volumes = append(podSpec.Volumes, volumes...)
	podSpec.ImagePullSecrets = append(podSpec.ImagePullSecrets, imagePullSecrets...)
	backend.UpdatePodSpec(&podSpec, numberOfNodes, role, component, serviceName)
	return podSpec, nil
}

func setMetricsLabels(labels map[string]string, dynamoGraphDeployment *v1alpha1.DynamoGraphDeployment) {
	// Convert user-provided metrics annotation into controller-managed label
	// By default (no annotation), metrics are enabled
	if metricsAnnotationValue, ok := dynamoGraphDeployment.Annotations[commonconsts.KubeAnnotationEnableMetrics]; ok && metricsAnnotationValue == commonconsts.KubeLabelValueFalse {
		// Explicitly disabled, don't add the label
		return
	}
	// Any other value (including empty) enables metrics
	labels[commonconsts.KubeLabelMetricsEnabled] = commonconsts.KubeLabelValueTrue
}

func generateComponentContext(component *v1alpha1.DynamoComponentDeploymentOverridesSpec, parentGraphDeploymentName string, namespace string, numberOfNodes int32) ComponentContext {
	componentContext := ComponentContext{
		numberOfNodes:                  numberOfNodes,
		ParentGraphDeploymentName:      parentGraphDeploymentName,
		ParentGraphDeploymentNamespace: namespace,
	}
	if component.DynamoNamespace != nil {
		componentContext.DynamoNamespace = *component.DynamoNamespace
	}
	return componentContext
}

// GeneratePodSpecForComponent creates a PodSpec for Grove deployments (simplified wrapper)
func GeneratePodSpecForComponent(
	component *v1alpha1.DynamoComponentDeploymentOverridesSpec,
	backendFramework BackendFramework,
	secretsRetriever SecretsRetriever,
	dynamoDeployment *v1alpha1.DynamoGraphDeployment,
	role Role,
	numberOfNodes int32,
	controllerConfig controller_common.Config,
	multinodeDeploymentType commonconsts.MultinodeDeploymentType,
	serviceName string,
) (corev1.PodSpec, error) {
	if len(dynamoDeployment.Spec.Envs) > 0 {
		component.Envs = MergeEnvs(dynamoDeployment.Spec.Envs, component.Envs)
	}
	podSpec, err := GenerateBasePodSpec(component, backendFramework, secretsRetriever, dynamoDeployment.Name, dynamoDeployment.Namespace, role, numberOfNodes, controllerConfig, multinodeDeploymentType, serviceName)
	if err != nil {
		return corev1.PodSpec{}, err
	}
	return podSpec, nil
}

// GenerateGrovePodGangSet generates a Grove PodGangSet for the given deployment, supporting both single-node and multinode cases.
func GenerateGrovePodGangSet(
	ctx context.Context,
	dynamoDeployment *v1alpha1.DynamoGraphDeployment,
	controllerConfig controller_common.Config,
	secretsRetriever SecretsRetriever,
) (*grovev1alpha1.PodGangSet, error) {
	gangSet := &grovev1alpha1.PodGangSet{}
	gangSet.Name = dynamoDeployment.Name
	gangSet.Namespace = dynamoDeployment.Namespace
	gangSet.Spec.Replicas = 1
	gangSet.Spec.Template.HeadlessServiceConfig = &grovev1alpha1.HeadlessServiceConfig{
		PublishNotReadyAddresses: true,
	}
	if controllerConfig.Grove.TerminationDelay > 0 {
		gangSet.Spec.Template.TerminationDelay = &metav1.Duration{Duration: controllerConfig.Grove.TerminationDelay}
	}
	dynamoNamespace, err := getDynamoNamespace(dynamoDeployment)
	if err != nil {
		return nil, fmt.Errorf("failed to get the graph dynamo namespace: %w", err)
	}
	var scalingGroups []grovev1alpha1.PodCliqueScalingGroupConfig
	for serviceName, component := range dynamoDeployment.Spec.Services {
		component.DynamoNamespace = &dynamoNamespace
		// Determine backend framework using hybrid approach
		backendFramework, err := getBackendFrameworkFromComponent(component, dynamoDeployment)
		if err != nil {
			return nil, fmt.Errorf("failed to determine backend framework for service %s: %w", serviceName, err)
		}

		numberOfNodes := component.GetNumberOfNodes()
		isMultinode := numberOfNodes > 1
		roles := expandRolesForService(serviceName, component.Replicas, numberOfNodes)
		var cliqueNames []string

		for _, r := range roles {
			podSpec, err := GeneratePodSpecForComponent(
				component,
				backendFramework,
				secretsRetriever,
				dynamoDeployment,
				r.Role,
				numberOfNodes,
				controllerConfig,
				commonconsts.MultinodeDeploymentTypeGrove,
				serviceName,
			)
			if err != nil {
				return nil, fmt.Errorf("failed to generate podSpec for role %s: %w", r.Name, err)
			}

			clique := &grovev1alpha1.PodCliqueTemplateSpec{
				Name: strings.ToLower(r.Name),
				Spec: grovev1alpha1.PodCliqueSpec{
					RoleName: strings.ToLower(r.Name),
					Replicas: r.Replicas,
					PodSpec:  podSpec,
				},
			}
			labels, err := generateLabels(component, dynamoDeployment, r.Name)
			if err != nil {
				return nil, fmt.Errorf("failed to generate labels: %w", err)
			}
			clique.Labels = labels
			annotations, err := generateAnnotations(component)
			if err != nil {
				return nil, fmt.Errorf("failed to generate annotations: %w", err)
			}
			clique.Annotations = annotations
			gangSet.Spec.Template.Cliques = append(gangSet.Spec.Template.Cliques, clique)
			cliqueNames = append(cliqueNames, strings.ToLower(r.Name))
		}

		// Apply startup dependencies for this service
		applyCliqueStartupDependencies(gangSet, roles, backendFramework, numberOfNodes)

		if isMultinode {
			scalingGroups = append(scalingGroups, grovev1alpha1.PodCliqueScalingGroupConfig{
				Name:        strings.ToLower(serviceName),
				CliqueNames: cliqueNames,
				Replicas:    component.Replicas,
			})
		}
	}
	if len(scalingGroups) > 0 {
		gangSet.Spec.Template.PodCliqueScalingGroupConfigs = scalingGroups
	}

	return gangSet, nil
}

func generateLabels(component *v1alpha1.DynamoComponentDeploymentOverridesSpec, dynamoDeployment *v1alpha1.DynamoGraphDeployment, componentName string) (map[string]string, error) {
	labels := make(map[string]string)
	labels[commonconsts.KubeLabelDynamoSelector] = GetDynamoComponentName(dynamoDeployment, componentName)
	if component.ComponentType != "" {
		labels[commonconsts.KubeLabelDynamoComponentType] = component.ComponentType
	}
	setMetricsLabels(labels, dynamoDeployment)
	if component.Labels != nil {
		err := mergo.Merge(&labels, component.Labels, mergo.WithOverride)
		if err != nil {
			return nil, fmt.Errorf("failed to merge labels: %w", err)
		}
	}
	if component.ExtraPodMetadata != nil {
		err := mergo.Merge(&labels, component.ExtraPodMetadata.Labels, mergo.WithOverride)
		if err != nil {
			return nil, fmt.Errorf("failed to merge extraPodMetadata labels: %w", err)
		}
	}
	return labels, nil
}

func generateAnnotations(component *v1alpha1.DynamoComponentDeploymentOverridesSpec) (map[string]string, error) {
	annotations := make(map[string]string)
	if component.Annotations != nil {
		err := mergo.Merge(&annotations, component.Annotations, mergo.WithOverride)
		if err != nil {
			return nil, fmt.Errorf("failed to merge annotations: %w", err)
		}
	}
	if component.ExtraPodMetadata != nil {
		err := mergo.Merge(&annotations, component.ExtraPodMetadata.Annotations, mergo.WithOverride)
		if err != nil {
			return nil, fmt.Errorf("failed to merge extraPodMetadata annotations: %w", err)
		}
	}
	return annotations, nil
}

// detectBackendFrameworkFromArgs detects the backend framework from command/args
func detectBackendFrameworkFromArgs(command []string, args []string) (BackendFramework, error) {
	// Combine command and args to search through all parts
	allParts := append(command, args...)
	fullCommand := strings.Join(allParts, " ")

	// Pattern to match python -m dynamo.{backend}.something
	patterns := map[BackendFramework]*regexp.Regexp{
		BackendFrameworkVLLM:   regexp.MustCompile(`python[0-9.]*\s+[^|&;]*-m\s+[^|&;]*dynamo\.vllm[^|&;]*`),
		BackendFrameworkSGLang: regexp.MustCompile(`python[0-9.]*\s+[^|&;]*-m\s+[^|&;]*dynamo\.sglang[^|&;]*`),
		BackendFrameworkTRTLLM: regexp.MustCompile(`python[0-9.]*\s+[^|&;]*-m\s+[^|&;]*dynamo\.trtllm[^|&;]*`),
	}

	var detected []BackendFramework
	for framework, pattern := range patterns {
		if pattern.MatchString(fullCommand) {
			detected = append(detected, framework)
		}
	}

	if len(detected) == 0 {
		return "", fmt.Errorf("no backend framework detected from command: %q", fullCommand)
	}

	if len(detected) > 1 {
		return "", fmt.Errorf("multiple backend frameworks detected from command: %v in %q", detected, fullCommand)
	}

	return detected[0], nil
}

// BackendFrameworkNoop represents no backend processing needed
const BackendFrameworkNoop BackendFramework = "noop"

// determineBackendFramework is the core logic for hybrid backend framework detection
// Takes extracted parameters and applies the detection logic
func determineBackendFramework(
	componentType string,
	command []string,
	args []string,
	explicitBackendFramework string,
) (BackendFramework, error) {
	// Check if this is a worker component - if not, use noop backend
	if !isWorkerComponent(componentType) {
		return BackendFrameworkNoop, nil
	}

	// Worker component - apply backend framework detection
	var detectedFramework BackendFramework
	var detectionError error

	// Try to detect from command/args
	if len(command) > 0 || len(args) > 0 {
		detected, err := detectBackendFrameworkFromArgs(command, args)
		if err == nil {
			detectedFramework = detected
		} else {
			detectionError = err
		}
	}

	// Get explicit framework
	var explicitFramework BackendFramework
	if explicitBackendFramework != "" {
		explicitFramework = BackendFramework(explicitBackendFramework)
	}

	// Validate consistency if both detected and explicit exist
	if detectedFramework != "" && explicitFramework != "" && detectedFramework != explicitFramework {
		return "", fmt.Errorf("backend framework mismatch: detected %q from command but explicitly configured as %q",
			detectedFramework, explicitFramework)
	}

	// Return in order of preference: detected > explicit > error
	if detectedFramework != "" {
		return detectedFramework, nil
	}

	if explicitFramework != "" {
		return explicitFramework, nil
	}

	// If we couldn't detect and no explicit config, return error
	if detectionError != nil {
		return "", fmt.Errorf("could not determine backend framework: %w", detectionError)
	}

	// No command/args to detect from and no explicit config
	return "", fmt.Errorf("backend framework must be specified explicitly or detectable from command/args")
}

// getBackendFrameworkFromComponent attempts to determine backend framework using hybrid approach:
// 1. Check if component is a worker - if not, return noop
// 2. For workers: try to detect from command/args, fall back to explicit config
// 3. Return error if worker has neither detection nor explicit config
// Also validates consistency between detected and explicit if both exist
func getBackendFrameworkFromComponent(
	component *v1alpha1.DynamoComponentDeploymentOverridesSpec,
	dynamoDeployment *v1alpha1.DynamoGraphDeployment,
) (BackendFramework, error) {
	// Extract command/args from component
	var command, args []string
	if component.ExtraPodSpec != nil && component.ExtraPodSpec.MainContainer != nil {
		command = component.ExtraPodSpec.MainContainer.Command
		args = component.ExtraPodSpec.MainContainer.Args
	}

	// Extract explicit backend framework from deployment
	explicitBackendFramework := dynamoDeployment.Spec.BackendFramework

	return determineBackendFramework(
		component.ComponentType,
		command,
		args,
		explicitBackendFramework,
	)
}

// ConvertDynamoComponentDeploymentToSpec converts a DynamoComponentDeployment to our component spec interface
// This is a helper for the controller to use our backend logic
func ConvertDynamoComponentDeploymentToSpec(dynComponent *v1alpha1.DynamoComponentDeployment) *v1alpha1.DynamoComponentDeploymentOverridesSpec {
	return &v1alpha1.DynamoComponentDeploymentOverridesSpec{
		DynamoComponentDeploymentSharedSpec: *dynComponent.Spec.DynamoComponentDeploymentSharedSpec.DeepCopy(),
	}
}

// getBackendFrameworkFromDynamoComponent determines backend framework for a DynamoComponentDeployment
func getBackendFrameworkFromDynamoComponent(dynComponent *v1alpha1.DynamoComponentDeployment) (BackendFramework, error) {
	// Extract command/args from component
	var command, args []string
	if dynComponent.Spec.ExtraPodSpec != nil && dynComponent.Spec.ExtraPodSpec.MainContainer != nil {
		command = dynComponent.Spec.ExtraPodSpec.MainContainer.Command
		args = dynComponent.Spec.ExtraPodSpec.MainContainer.Args
	}

	// Extract explicit backend framework
	explicitBackendFramework := dynComponent.Spec.BackendFramework

	return determineBackendFramework(
		dynComponent.Spec.ComponentType,
		command,
		args,
		explicitBackendFramework,
	)
}

// GenerateBasePodSpecForController generates a PodSpec using backend logic for controller usage
// This preserves the base pod generation while allowing controller-specific enhancements
func GenerateBasePodSpecForController(
	dynComponent *v1alpha1.DynamoComponentDeployment,
	secretsRetriever SecretsRetriever,
	controllerConfig controller_common.Config,
	role Role,
	multinodeDeploymentType commonconsts.MultinodeDeploymentType,
) (corev1.PodSpec, error) {
	// Convert to our interface
	componentSpec := ConvertDynamoComponentDeploymentToSpec(dynComponent)

	numberOfNodes := componentSpec.GetNumberOfNodes()

	// Determine backend framework using hybrid approach
	backendFramework, err := getBackendFrameworkFromDynamoComponent(dynComponent)
	if err != nil {
		return corev1.PodSpec{}, fmt.Errorf("failed to determine backend framework: %w", err)
	}

	// Generate base PodSpec with standard env vars using merged component envs
	// For controller usage, we may not have serviceName, so use the component name as fallback
	serviceName := dynComponent.Name
	podSpec, err := GenerateBasePodSpec(
		componentSpec,
		backendFramework,
		secretsRetriever,
		dynComponent.GetParentGraphDeploymentName(),
		dynComponent.Namespace,
		role,
		numberOfNodes,
		controllerConfig,
		multinodeDeploymentType,
		serviceName,
	)
	if err != nil {
		return corev1.PodSpec{}, err
	}

	return podSpec, nil
}

func generateSharedMemoryVolumeAndMount(resources *corev1.ResourceRequirements) (corev1.Volume, corev1.VolumeMount) {
	sharedMemorySizeLimit := resource.MustParse("512Mi")
	// Check if we have memory limits to work with
	memoryLimit := resources.Limits[corev1.ResourceMemory]
	if !memoryLimit.IsZero() {
		// Use 1/4 of memory limit
		calculatedSize := resource.NewQuantity(memoryLimit.Value()/4, resource.BinarySI)
		// Apply bounds: minimum 512Mi, maximum 8Gi
		minSize := resource.MustParse("512Mi")
		maxSize := resource.MustParse("8Gi")

		if calculatedSize.Cmp(minSize) > 0 && calculatedSize.Cmp(maxSize) < 0 {
			sharedMemorySizeLimit = *calculatedSize
		} else if calculatedSize.Cmp(maxSize) >= 0 {
			sharedMemorySizeLimit = maxSize // Cap at maximum
		}
		// If calculatedSize < minSize, keep the 512Mi base
	}
	volume := corev1.Volume{
		Name: commonconsts.KubeValueNameSharedMemory,
		VolumeSource: corev1.VolumeSource{
			EmptyDir: &corev1.EmptyDirVolumeSource{
				Medium:    corev1.StorageMediumMemory,
				SizeLimit: &sharedMemorySizeLimit,
			},
		},
	}
	volumeMount := corev1.VolumeMount{
		Name:      commonconsts.KubeValueNameSharedMemory,
		MountPath: "/dev/shm",
	}
	return volume, volumeMount
}

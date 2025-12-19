#!/bin/bash

# install Spark
helm repo add bitnami https://charts.bitnami.com/bitnami --force-update
helm upgrade spark bitnami/spark --namespace spark --version 10.0.3 --values ./config/spark-values.yaml --install --create-namespace

# install Postgres with pgvector extension
#
# https://cloudnative-pg.io/docs/1.28/quickstart
helm repo add cnpg https://cloudnative-pg.github.io/charts --force-update
helm upgrade --install \
    cnpg cnpg/cloudnative-pg \
    --namespace cnpg-system --create-namespace \
    --version 0.27.0

kubectl get namespace postgres || kubectl create namespace postgres
kubectl --namespace postgres apply -f ./config/postgres.yaml
# Get postgres password by running:
#   kubectl -n postgres get secret marketplace-app -o jsonpath='{.data.password}' | base64 -d

# install Prometheus + Grafana for monitoring
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts --force-update
helm upgrade --install \
  prometheus-community prometheus-community/kube-prometheus-stack \
  --namespace prometheus --create-namespace \
  --values https://raw.githubusercontent.com/cloudnative-pg/cloudnative-pg/main/docs/src/samples/monitoring/kube-stack-config.yaml
  --version 80.13.3
# Get Grafana 'admin' user password by running:
#   kubectl --namespace prometheus get secrets prometheus-community-grafana -o jsonpath="{.data.admin-password}" | base64 -d ; echo
# Access Grafana local instance:
#   export POD_NAME=$(kubectl --namespace prometheus get pod -l "app.kubernetes.io/name=grafana,app.kubernetes.io/instance=prometheus-community" -oname)
#   kubectl --namespace prometheus port-forward $POD_NAME 3000

# expose services
kubectl apply -f ./config/expose.yaml



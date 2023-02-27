import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_predictions(train_data,
                      train_labels,
                      test_data,
                      test_labels,
                      p_train=None,
                      p_test=None):
  """
  Plots training data,test data and compare pred.
  """
  plt.figure(figsize=(10,7))
  #plot training data
  plt.scatter(train_data,train_labels,c="b",s=4,label="Training data")
  
  #plot test data
  plt.scatter(test_data,test_labels,c="g",s=4,label="Test data")

  if p_train is not None:
    plt.title("Training predictions")
    plt.scatter(train_data,p_train,c="r",s=4,label="Predictions")

  #Are there preds?
  if p_test is not None:
    plt.title("Test predictions")
    plt.scatter(test_data,p_test,c="r",s=4,label="Predictions")

  #Show the legend
  plt.legend(prop={"size":14})   


def lossfn_curve(loss,epochs):
  """
  Plots the loss curve of the data
  """
  steps=[i for i in range(epochs)]
  plt.plot(list(steps),loss)
  plt.xlabel("steps")
  plt.ylabel("Loss")
  plt.title("Loss_fn Curve")


def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.
    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

def plot_classfication_data_V0(X):
  """"
  Plots the make_circle dataset,
  First custom hf made by me a.k.a Manan!
  """
  plt.scatter(x=X[:,0],y=X[:,1],c=y,cmap=plt.cm.RdYlBu);


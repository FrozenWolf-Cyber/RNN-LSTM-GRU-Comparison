# RNN-LSTM-GRU-Comparison 

We are trying to compare train losses, training time, and model convergence for RNN , LSTM, and GRU models using sin(x) graph in CPU and GPU

## Result :
### RNN :
![rnn_final](https://user-images.githubusercontent.com/57902078/140617500-f8a122f4-1965-4b1a-b5a9-87c430df8430.png)

### LSTM :
![lstm_final](https://user-images.githubusercontent.com/57902078/140617511-d8c03b69-3c79-4dcd-b379-bd0dff4c790f.png)

### GRU :
![gnu_final](https://user-images.githubusercontent.com/57902078/140617519-4c862c9e-4ce9-45d5-8a04-591ac8bf1732.png)

### Comparison :
#### GPU :
![avg_training_time](https://user-images.githubusercontent.com/57902078/140617552-765cc071-0c51-4788-a7be-fe9d098671aa.png)![avg_testing_time](https://user-images.githubusercontent.com/57902078/140617556-444c112a-f940-44d7-b379-1e61e8df5d55.png)

#### CPU :
![avg_training_time_cpu](https://user-images.githubusercontent.com/57902078/140617581-0c8ffdd4-5bb2-4870-bc2d-23ad4d5e9c09.png)![avg_testing_time_cpu](https://user-images.githubusercontent.com/57902078/140617584-76fb0977-2dbe-4022-ab00-020683c8d56d.png)

#### Loss :
![loss_epoch](https://user-images.githubusercontent.com/57902078/140617651-65bc30b7-f5b7-42a1-961b-a091f7740bda.png)![loss_minibatch](https://user-images.githubusercontent.com/57902078/140617653-1ac8b67e-9e94-4db3-960e-d69c76b9c83a.png)

#### Predictions :
![epoch5](https://user-images.githubusercontent.com/57902078/140617668-07c19c42-aba4-41fd-a949-2812ba3fe8ef.png)
![epoch10](https://user-images.githubusercontent.com/57902078/140617670-4694d8ff-67f0-4445-ad12-a67f510d0d43.png)
![epoch15](https://user-images.githubusercontent.com/57902078/140617673-9bd43637-75b9-4f25-aece-3b9dc9a1fde6.png)
![epoch20](https://user-images.githubusercontent.com/57902078/140617674-85258ef5-60d4-4a24-9ca2-d86228d50c68.png)
![epoch25](https://user-images.githubusercontent.com/57902078/140617678-ca095725-97c6-4385-97f8-9a631a9816a0.png)
![epoch30](https://user-images.githubusercontent.com/57902078/140617680-dbcdb45f-1a1d-4e45-967e-0e903aaa5c77.png)
![epoch35](https://user-images.githubusercontent.com/57902078/140617689-09b557dc-6ee5-4dc5-a800-d9e3afa9674e.png)
![epoch40](https://user-images.githubusercontent.com/57902078/140617692-1bc5294d-ef74-4013-983e-653b163f93b5.png)
![epoch45](https://user-images.githubusercontent.com/57902078/140617693-f7aa5511-25ac-4bca-ae6d-ca9977d1313a.png)
![epoch50](https://user-images.githubusercontent.com/57902078/140617694-79b30aa2-d1b6-4a3c-b5c6-86019323ffba.png)

## Conclusion :
From the results, we can conclude that it's easier to tweak RNN and GRU than LSTM due to the larger number of gates. It is clear that GRU is much faster than LSTM and GRU during training due to its very low number of trainable parameters. It is also clear that LSTM takes a longer time to converge than the other two models.



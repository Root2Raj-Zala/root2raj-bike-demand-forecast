# Decision brief

Question: after yesterday's rental count becomes available, how useful is a next-day forecast compared with a simple recent-count rule?

The selected model missed by 775 rentals per day on average and improved MAE by 7.8% over the validation-selected baseline. The small improvement must be weighed against added complexity. The residual band missed more actual values than its nominal level suggests.

Recommended next experiment: collect a current rolling evaluation and compare errors during high-demand days, weather changes and system expansion. An operations team should define the real costs of over- and underplanning before choosing a decision threshold. The current evidence cannot justify a fleet size or quantify savings.

Interview explanation: I used actual historical records. I kept the future out of feature construction, chose the model on earlier dates and reported performance only after freezing that choice. I can explain those decisions; the implementation was AI-assisted.

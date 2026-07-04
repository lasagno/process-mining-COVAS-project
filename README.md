<h1>Process Mining by Age Group</h1>

Discovers and evaluates process models from a combined event log (XES), split by patient/subject age group, using four different process discovery algorithms. Each python script partitions the log into age-decade buckets (e.g. 40-49, 50-59, ...) plus a pooled Overall group, discovers a Petri net for every group, model simplicity scores and cross-evaluates every log against every model (fitness & precision).

<h2>Repository Structure</h2>
<ul>
  <li>Data/: Input event log(s), e.g. all_waves_merged.xes</li>
  <li>Results/: Generated outputs (Petri nets, matrices, plots)</li>
  <li>main_alpha.py: Discovery via Alpha Miner</li>
  <li>main_heuristic.py: Discovery via Heuristics Miner</li>
  <li>main_ilp.py: Discovery via ILP Miner</li>
  <li>main_inductive.py: Discovery via Inductive Miner</li>
</ul>


<h2>Code</h2>
<ol>
  <li>Load & prepare data: reads the .xes event log, detects an age attribute column and buckets cases into 10-year age groups.</li>
  <li>Add an "Overall" group: pools all cases together so each age group's model can be compared against an all-ages model</li>
  <li>Discover process models: runs the corresponding miner (Alpha / Heuristics / ILP / Inductive) on each age group's sublog to produce a Petri net per group</li>
  <li>Simplicity evaluation: compute the arc-degree simplicity for each discovered model into a bar chart</li>
  <li>Cross-evaluation: replay every age group's log against every discovered model, producing fitness and precision matrices with heatmaps</li>
</ol>

<h2>Outputs</h2>
<ul>
  <li>petri_net_<group>.png: visualized Petri net for each age group</li>
  <li>simplicity_metrics.csv, simplicity_chart.png: arc-degree simplicity per group</li>
  <li>fitness_matrix.csv, fitness_matrix_heatmap.png: cross-evaluation fitness</li>
  <li>precision_matrix.csv, precision_matrix_heatmap.png: cross-evaluation precision</li>
</ul>

<h2>Requirements</h2>
Python 3.x
pm4py, pandas, numpy, matplotlib, seaborn

<h3>Install with:</h3>
<code>pip install pm4py pandas numpy matplotlib seaborn</code>

<h2>Usage</h2>
Place event log in Data/all_waves_merged.xes or change the XES_FILE path in the code, then run the corresponding code:
<code>python main_alpha.py</code>
<code>python main_heuristic.py</code>
<code>python main_ilp.py</code>
<code>python main_inductive.py</code>
Results are written in Results/

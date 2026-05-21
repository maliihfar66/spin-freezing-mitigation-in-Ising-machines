clear; clc; close all;

%% ---------------- PARAMETERS ----------------
AC       = 1;                 %Coupling strength 
As0      = 3.5;        %SHI mean amplitude
m        = 100;      %number of oscillators
N        = 50;      % number of graphs
nTrials  = 10;      % number of simulations per graph

nSweeps  = 3e5;        %gibbs sampling steps
beta     = linspace(0.8,2,nSweeps); %inverse-temperature schedule
An       = 0.001;     %noise
tstop    = 10;        %simulation time
dt       = 2e-3;
Nsteps   = round(tstop/dt); %number of simulation steps




%% ---------------- RANDOM GRAPHS ----------------
p = [0.03 0.4 0.5 0.6 0.8];  %graph densities
w_idx = 1;
idxAll   = 1:m;
J = zeros(length(p),N,m,m);
s0 = [];
for w = 1:length(p)
    for k = 1:N
        for i = 1:m-1
            for j = i+1:m
                if rand <= p(w)
                    J(w,k,i,j) = -1;
                    J(w,k,j,i) = -1;
                end
            end
        end
        J(w,k,1:m+1:end) = 0;
    end
end

%% ---------------- UNIFORM Ks HETEROGENEITY ----------------
a = 0;     %min of ks value 
b = 7;     %max of ks value

ks_scale = a + (b-a)*rand(m,1);
ks_scale = ks_scale / mean(ks_scale);
As_vec   = As0 * ks_scale;

%% ---------------- PULSE FUNCTION ----------------
Tper = 0.02;   %pulse period
W    = 0.01;   %pulse width
t0   = 10e-3;   %pulse start time
pulse = @(t) double((t>=t0) & (mod(t-t0,Tper) < W));

%% ---------------- STORAGE ----------------
MC_raw  = zeros(N,nTrials,3);
normVal = zeros(N,nTrials,4);
Q_all   = zeros(N,1);

CUT_all = cell(1,3);
RHO_all = cell(1,3);

for mode = 1:3
    CUT_all{mode} = nan(Nsteps+1,N,nTrials);
    RHO_all{mode} = nan(Nsteps+1,N,nTrials);
end

last_T1 = [];
last_S1 = [];
last_Jn = [];
last_r = [];
last_cuts1 = [];

T_mode = cell(1,3);
S_mode = cell(1,3);
R_mode = cell(1,3);

mode_labels = {'homogeneous','hetero-static','hetero-pulsed'};

%% ---------------- MAIN LOOP ----------------
for n = 1:N

    Jn = squeeze(J(w_idx,n,:,:));
    Jn(1:m+1:end) = 0;

    %% Gibbs baseline
    s_gibbs = mcmc_gibbs_spins(Jn, beta, nSweeps, s0);
    Q = cut_from_spins(Jn, s_gibbs);
    Q_all(n) = Q;

    theta0 = ones(m,1);

    for mode = 1:3

        switch mode
            case 1
                AS_of_t = @(t) As0 * ones(m,1);

            case 2
                AS_of_t = @(t) As_vec;

            case 3
                AS_of_t = @(t) As_vec .* pulse(t);
        end

        driftFun = @(t,x) Kuramoto_AS(x, AC, AS_of_t(t), Jn);
        diffFun  = @(t,x) An * eye(m);

        for l = 1:nTrials

            mdl = sde(driftFun, diffFun, 'StartState', theta0);

            [Xsim,T1] = simByEuler(mdl, Nsteps, ...
                'DeltaTime', dt, ...
                'nTrials', 1);

            S1 = squeeze(Xsim(:,:,1));
            L  = size(S1,1);

            cuts1 = zeros(L,1);
            rmask = false(L,m);

            for tt = 1:L

                row = S1(tt,:);

                %% Cut value
                ix = find(mod(round(row),2));

                if isempty(ix) || numel(ix) == m
                    cuts1(tt) = 0;
                else
                    cuts1(tt) = -sum(sum(Jn(ix,setdiff(idxAll,ix))));
                end

                %% Freezing mask
                AS_vec_t  = AS_of_t(T1(tt));
                self_term = AS_vec_t(:).' .* sin(2*pi*row);

                SinMat    = sin(pi*(row.' - row));
                coup_term = sum(Jn .* SinMat,2).';

                rmask(tt,:) = (abs(coup_term) < abs(self_term)) & ...
                              ((coup_term .* self_term) < 0);
            end

            MC_raw(n,l,mode) = max(cuts1);

            CUT_all{mode}(1:L,n,l) = cuts1;
            RHO_all{mode}(1:L,n,l) = mean(rmask,2);

       
            if n == N && l == nTrials
                T_mode{mode} = T1;
                S_mode{mode} = S1;
                R_mode{mode} = rmask;

                if mode == 3
                    last_T1 = T1;
                    last_S1 = S1;
                    last_Jn = Jn;
                    last_r = rmask;
                    last_cuts1 = cuts1;
                end
            end
        end
    end

   
    for l = 1:nTrials

        Pm = squeeze(MC_raw(n,l,:)).';

        theta = max([Pm, Q]);
        if theta == 0
            theta = 1;
        end

        normVal(n,l,1:3) = Pm ./ theta;
        normVal(n,l,4)   = Q ./ theta;
    end
end


normHom = reshape(normVal(:,:,1),[],1);
normHet = reshape(normVal(:,:,2),[],1);
normPul = reshape(normVal(:,:,3),[],1);

rawHom = reshape(MC_raw(:,:,1),[],1);
rawHet = reshape(MC_raw(:,:,2),[],1);
rawPul = reshape(MC_raw(:,:,3),[],1);


disp(['Average raw homogeneous max cut           = ', num2str(mean(rawHom))])
disp(['Average raw heterogeneous max cut         = ', num2str(mean(rawHet))])
disp(['Average raw pulsed heterogeneous max cut  = ', num2str(mean(rawPul))])

disp(['Average normalized homogeneous            = ', num2str(mean(normHom))])
disp(['Average normalized heterogeneous          = ', num2str(mean(normHet))])
disp(['Average normalized pulsed hetero          = ', num2str(mean(normPul))])


edges = linspace(0,1,51);

figure('Name','All modes together normalized','Color','w');
hold on;

h1 = histogram(normHom, edges, 'DisplayName','homogeneous');
h2 = histogram(normHet, edges, 'DisplayName','hetero-static');
h3 = histogram(normPul, edges, 'DisplayName','hetero-pulsed');

set([h1 h2 h3], 'FaceAlpha',0.5, 'EdgeColor','none');

xlabel('Normalized max cut');
ylabel('Number of runs');
title(sprintf('Normalized cut over %d graphs × %d trials', N, nTrials));
legend('Location','northwest');
xlim([0 1]);
grid on;
box on;
hold off;

%% ---------------- RAW MAX CUT HISTOGRAM: ALL GRAPHS × ALL TRIALS ----------------
figure;
hold on;

histogram(rawHom, 'DisplayName','homogeneous');
histogram(rawHet, 'DisplayName','hetero-static');
histogram(rawPul, 'DisplayName','hetero-pulsed');

xlabel('Raw max cut');
ylabel('Number of runs');
title(sprintf('Raw max cut over %d graphs × %d trials', N, nTrials));
legend('Location','best');
grid on;
box on;
hold off;

%% ---------------- ERROR BAR PLOT ----------------
figure;
bar_vals = [mean(normHom), mean(normHet), mean(normPul)];
bar_errs = [std(normHom),  std(normHet),  std(normPul)];

bar(bar_vals);
hold on;
errorbar(1:3, bar_vals, bar_errs, '.k', 'LineWidth',1.5);

set(gca,'XTick',1:3, ...
    'XTickLabel',{'homogeneous','hetero-static','hetero-pulsed'});

ylabel('Normalized score');
title('Average normalized performance over all graphs and trials');
ylim([0 1.1]);
grid on;
box on;
hold off;

%% ---------------- PHASE PLOT REPRESENTATIVE RUN ----------------
if ~isempty(last_S1)
    figure;
    plot(last_T1,last_S1,'LineWidth',0.8);
    xlabel('Time');
    ylabel('phase');
    title('Representative phase trajectories');
    grid on;
end

%% ---------------- CUT VS TIME REPRESENTATIVE RUN ----------------
if ~isempty(last_cuts1)
    figure;
    plot(last_T1,last_cuts1,'LineWidth',1.2);
    xlabel('Time');
    ylabel('Cut value');
    title('Cut vs time, representative run');
    grid on;
end

%% ---------------- FREEZING HEATMAPS ----------------
for mode = 1:3
    if isempty(R_mode{mode})
        continue;
    end

    figure;
    M = double(R_mode{mode}.');
    h = heatmap(M);
    h.GridVisible = 'off';
    h.Colormap = cool;
    h.XDisplayLabels = repmat("",1,size(M,2));
    h.YDisplayLabels = repmat("",size(M,1),1);
    title(sprintf('Frozen spins — mode = %s', mode_labels{mode}));
end

%% ---------------- FREEZING RATIO OVER TIME: ALL GRAPHS × TRIALS ----------------
Time = (0:Nsteps).' * dt;

mean_rho_homog  = mean(RHO_all{1}, [2 3], 'omitnan');
mean_rho_hstat  = mean(RHO_all{2}, [2 3], 'omitnan');
mean_rho_hpulse = mean(RHO_all{3}, [2 3], 'omitnan');

win = 200;

mean_rho_homog_sm  = movmean(mean_rho_homog,win);
mean_rho_hstat_sm  = movmean(mean_rho_hstat,win);
mean_rho_hpulse_sm = movmean(mean_rho_hpulse,win);

figure;
hold on;

plot(Time,mean_rho_homog_sm,'LineWidth',1.8,'DisplayName','homogeneous');
plot(Time,mean_rho_hstat_sm,'LineWidth',1.8,'DisplayName','hetero-static');
plot(Time,mean_rho_hpulse_sm,'LineWidth',1.8,'DisplayName','hetero-pulsed');

xlabel('Time');
ylabel('\rho');
title('Freezing ratio averaged over all graphs and trials');
ylim([0 1]);
legend('Location','best');
grid on;
box on;
hold off;

%% ---------------- MEAN CUT OVER TIME: ALL GRAPHS × TRIALS ----------------
mean_cut_homog  = mean(CUT_all{1}, [2 3], 'omitnan');
mean_cut_hstat  = mean(CUT_all{2}, [2 3], 'omitnan');
mean_cut_hpulse = mean(CUT_all{3}, [2 3], 'omitnan');

mean_cut_homog_sm  = movmean(mean_cut_homog,win);
mean_cut_hstat_sm  = movmean(mean_cut_hstat,win);
mean_cut_hpulse_sm = movmean(mean_cut_hpulse,win);

figure;
hold on;

plot(Time,mean_cut_homog_sm,'LineWidth',2,'DisplayName','homogeneous');
plot(Time,mean_cut_hstat_sm,'LineWidth',2,'DisplayName','hetero-static');
plot(Time,mean_cut_hpulse_sm,'LineWidth',2,'DisplayName','hetero-pulsed');

xlabel('Time');
ylabel('Mean cut');
title('Mean cut over time averaged over all graphs and trials');
legend('Location','best');
grid on;
box on;
hold off;
% %% ---------------- FIGURE: homogeneous dual-axis (rho + cut) ----------------
% figure;
% hold on;
% grid on;
% box on;
% 
% % LEFT axis → freezing ratio
% yyaxis left
% plot(Time, mean_rho_homog_sm, 'LineWidth', 2);
% ylabel('Freezing ratio');
% 
% % RIGHT axis → mean cut
% yyaxis right
% plot(Time, mean_cut_homog_sm, 'LineWidth', 2);
% ylabel('Mean cut');
% 
% xlabel('Time');
% title('Homogeneous: Freezing ratio vs Mean cut');
% 
% legend({'Freezing ratio','Mean cut'}, 'Location','best');

%% ---------------- FUNCTIONS ----------------
function f = Kuramoto_AS(x, AC, ASvec, J)

    x = x(:);

    S = sin(pi*(x - x.'));

    f = (-AC * sum(J .* S,2) - ASvec .* sin(2*pi*x)) / pi;
end

function s = mcmc_gibbs_spins(J,beta,nSweeps,s0)

    n = size(J,1);

    if nargin < 4 || isempty(s0)
        s = sign(randn(n,1));
        s(s==0) = 1;
    else
        s = s0(:);
        s(s==0) = 1;
    end

    for sweep = 1:nSweeps
        idx = randperm(n);

        for i = idx
            h_i = J(i,:)*s;

            if beta(sweep) <= 0
                p = 0.5;
            else
                p = 1 ./ (1 + exp(-2*beta(sweep)*h_i));
            end

            s(i) = 2*(rand < p) - 1;
        end
    end
end

function cutVal = cut_from_spins(J,s)

    A = (J == -1);

    cutVal = 0.5 * sum(sum(triu(A,1) .* (1 - (s*s.'))));
end
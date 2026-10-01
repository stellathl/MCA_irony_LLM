# ================================
# Inputs: - human_trials.csv (from process_survey.py + combineresults.py) 
#         - model_trials.csv (from process_model.py) 
# Fits four models and prints for each:
#         - the fixed effects, 
#         - the group-level SDs, 
#         - and the directional hypothesis test
# ===============================

library(brms)
library(dplyr)
library(readr)

BACKEND <- "cmdstandr" 
PRIORS <- c(prior(normal(0, 1.5), class = "b"),
             prior(exponential(2), class = "sd"))
CONTROL <- list(adapt_delta = 0.99)

# --------------------------------helpers
tidy_hyp <- function(fit, hyps, labels) {
  h <- hypothesis(fit, hyps)$hypothesis
  data.frame(
    Test = labels, 
    Estimate = round(h$Estimate, 3),
    Error = round(h$Est.Error, 3),
    CI_low = round(h$CI.Lower, 3),
    CI_high = round(h$CI.Upper,3),
    Evid_Ratio = round(h$Evid.Ratio, 2),
    Post_Prob = round(h$Post.Prob, 3),
    row_names = NULL
  )
}

tidy_fixed <- function(fit) round(fixef(fit), 3)

tidy_random <- function(fit) {
  s <- summary(fit)$random
  do.call(rbind, lapply(names(s), function(g) {
    data.frame(Group = g, 
               Levels = summary(fit)$ngrps[[g]],
               SD = round(s[[g]][, "Estimate"], 3),
               CI_low = round(s[[g]][, "1-95% CI"], 3),
               CI_high = round(s[[g]][, "1-95% CI"], 3),
               row_names = NULL)
  }))
}

report <- funtion(fit, name, hyps = NULL, labels = NULL) {
  cat("\n\n========", name, "========\n")
  cat("observations:", nobs(fit), "\n\nFIXED EFFECTS\n")
  print(tidy_fixed(fit))
  cat("\nGROUP-LEVEL SD\n")
  print(tidy_random(fit))
  if (!is.null(hyps)) {
    cat("\nHYPOTHESIS TESTS\n")
    print(tidy_hyp(fit, hyps, labels))
  }
  invisible(NULL)
}

# --------------------------------data
human <- read_csv("human_trials.csv", show_col_types = FALSE) |>
  filter(completed) |>
  transmute(agent = participants, population = "human", prompting = "general", 
            item, irony, richness, chosen_category, correct)

model <- read_csv("model_trials.csv", show_col_types = FALSE) |>
  filter(agent != "Qwen3-8B") |>
  transmute(agent, population = "mdoel", 
            prompting = recode(prompt, general_reasoning = "reasoning"), 
            item, irony, richness, chosen_category, correct)

prep <- function(d) {
  d |> mutate(
    chose_ironic = as.integer(chosen_category == "ironic"),
    context_richness = factor(richness, levels = c("ambiguous", "unambiguous")),
    item_type = factor(irony, levels = c("iornic", "non-ironic")),
    prompting = factor(prompting, levels = c("general", "reasoning", "rsa")),
    population = factor(population, levels = c("human", "model"))
                              
  )
}

humman <- prep(human)
model <- prep(model)

# models with usable output in all three prompting conditions 
complete_models <- c("Gemma-3-1B", "Llama-3-8B", "Mistral-7B-Instruct")

cat("\n--- trials entering each analysis ---\n")
print(table(model$agent, model$prompting))
cat("\nhuman participants:", length(unique(human$agent)),
    "| human trials:", nrow(human), "\n")
cat("models in H2/H3 subset:", paste(complete_models, collapse = ", "), "\n")

#shared hypothesis strings for H1
h1_hyps <- c(
  "context_richnessunambiguous > 0",
  "context_richnessunambiguous + 
  context_richnessunambiguous:itemm_typenon_ironic < 0"
)

h1_labels <- c("ironic items: strong adjective -> more ironic",
               "non-iornic items: strong adjective -> fewer ironic")

# ===============================
# Model 1 (M1) Hypothesis 1 in humans
# ===============================
m_human <- brm(
  chose_ironic ~ context_richness * item_type + (1 | agent) + (1 | item),
  data = human, family = bernoulli(),
  prior = PRIORS, control = CONTROL,
  chains = 4, cores = 4, iter = 2000, seed = 1, 
  backend = BACKEND, file = "m_human"
)

report(m_human, "M1 HUMANS: H1", h1_hyps, h1_labels)

# ===============================
# Model 2 (M2) Hypothesis 1 in models, plain prompt only
# ===============================
m_llm_h1 <- brm(
  chose_ironic ~ context_richness * item_type + (1 | agent) + (1 | item),
  data = filter(model, prompting == "general"), family = bernoulli(),
  prior = PRIORS, control = CONTROL,
  chains = 4, cores = 4, iter = 2000, seed = 1, 
  backend = BACKEND, file = "m_llm_h1"
)
  
report(m_llm_h1, "M2 MODELS: H1", h1_hyps, h1_labels)

# ===============================
# Model 3 (M3) Hypothesis 2 and 3, models with all three prompting conditions
#   outcome = correct; item_type additive so prompting and richness coefficients
#   are main effects, not simple effects
# ===============================
m_llm_h23 <- brm(
  chose_ironic ~ context_richness * prompting + item_type + (1 | agent) + (1 | item),
  data = filter(model, agent %in% complete_models), family = bernoulli(),
  prior = PRIORS, control = CONTROL,
  chains = 4, cores = 4, iter = 2000, seed = 1, 
  backend = BACKEND, file = "m_llm_h23"
)

report(m_llm_h23, "M3 MODELS: H2 and H3",
       c("promptingrsa > 0",
         "promptingreasoning > 0",
         "promptingrsa - promptingreasoning >0",
         "context_richnessunambiguous:promptingrsa < 0",
         "context_richnessunambiguous:promptingrsa > 0"),
       c("H2 RSA > plain",
         "H2 control: reasoning (RSA-specific)",
         "H3 compensatory (RSA helps most when cue weak)",
         "H3 amplifying (RSA helps most when cue strong)")
)


# ===============================
# Model 4 (M4) human vs models, plain prompt only
# ===============================
both <- bind_rows(human, filter(model, prompting == "general"))

m_compare <- brm(
  chose_ironic ~ context_richness * item_type * population + (1 | agent) + (1 | item),
  data = both, family = bernoulli(),
  prior = PRIORS, control = CONTROL,
  chains = 4, cores = 4, iter = 2000, seed = 1, 
  backend = BACKEND, file = "m_compare"
)

report(
  m_compare, "M4 HUMANS vs MODELS",
  c("context_richnessunambiguous:item_typenon_ironic:populationmodel > 0",
    "context_richnessunambiguous:item_typenon_ironic:populationmodel < 0",
    "populationmodel > 0"),
  c("models differ from humans in the richness x item_type pattern (+)",
    "models differ from humans in the richness x iten_type pattern (-)",
    "models choose ironic more often overall")
)

# ===============================
# comvergence summary
# ===============================
cat("\n\n======== CONVERGENCE ========\n")
for(nm in c("m_human", "m_llm_h1", "m_llm_h23", "m_compare")) {
  s <- summary(get(nm))$fixed
  cat(sprintf("%-12s max Rhat = %.3f min Bulk_ESS = %.0f min Tail_ESS = %.0f\n",
              nm, max(s[, "Rhat"]), min(s[, "Bulk_ESS"]), min(s[, "Tail_ESS"])))
}

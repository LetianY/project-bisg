library(wru)
library(dplyr)
future::plan(future::multisession)
Sys.setenv(CENSUS_API_KEY = "YOUR_CENSUS_API_KEY")

voters = readRDS("nc_voter_reg_cleanedwbisg.rds")
# print(head(voters, 5)) # print the first 10 rows of the data
print(colnames(voters)) # print the columns of the data
print(dim(voters)) # print the number of rows and columns of the data

# rename column zcta5 to zcta
voters_filtered <- voters %>% select(race_desc, surname, zcta5, state, county) %>% rename(zcta = zcta5)
print(colnames(voters_filtered))

# predict_race(voter.file = voters, surname.only = TRUE)
result = predict_race(
    voter.file=voters_filtered, 
    census.surname = TRUE, 
    surname.only = FALSE,
    census.geo = "zcta",
    year = "2020",
    census.key = Sys.getenv("CENSUS_API_KEY"),
    census.data = get_census_data(
        year = "2020",
        key = Sys.getenv("CENSUS_API_KEY"),
        states = "NC",
        census.geo = "zcta",
        age = FALSE,
        sex = FALSE,
        retry = 3,
        county.list = NULL
    ),
    model = "BISG",
    names.to.use = 'surname',
    age = FALSE,
    sex = FALSE,
    retry = 3,
    impute.missing = FALSE,
    skip_bad_geos = TRUE,
    use.counties = FALSE,
    # party = NULL,
    # race.init,
    # name.dictionaries,
    )

print(result)

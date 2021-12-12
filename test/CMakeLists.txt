add_executable(GPTest
    main.cpp
)

target_link_libraries(GPTest
    PRIVATE
        GPModel::GPModel
)


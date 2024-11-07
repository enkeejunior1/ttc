import random
import os

def generate_n_by_n(order):
    # Randomly sample two numbers
    multiplicand = random.randint(10**(order-1), 10**order)  # Adjust the range as needed
    multiplier = random.randint(10**(order-1), 10**order)    # Adjust the range as needed

    # Compute the product
    product = multiplicand * multiplier

    # Extract digits of the multiplier in reverse order
    multiplier_digits = [int(d) for d in str(multiplier)][::-1]

    partial_products = []
    cumulative_sum = 0
    cumulative_sums = []

    for i, digit in enumerate(multiplier_digits):
        # partial_product for display
        partial_product = multiplicand * digit * 10**i
        partial_product_rev = str(partial_product)[::-1]
        partial_product_rev = partial_product_rev + '0' * max(0, order+1+i-len(partial_product_rev))
        partial_products.append(partial_product_rev)

        # Update cumulative sum
        cumulative_sum += partial_product

        # Reverse cumulative sum for display
        cumulative_sum_rev = str(cumulative_sum)[::-1]
        cumulative_sum_rev = cumulative_sum_rev + '0' * max(0, order+1+i-len(cumulative_sum_rev))
        cumulative_sums.append(cumulative_sum_rev)

    # Format the chain of thought
    cot_str = [' '.join(list(partial_products[0]))]
    for pp, cs in zip(partial_products[1:-1], cumulative_sums[1:-1]):
        pp_str = ' '.join([' '.join(list(pp)), '( ' + ' '.join(list(cs)) + ' )'])
        cot_str.append(pp_str)
    cot_str.append(' '.join(list(partial_products[-1])))
    cot_str = ' + '.join(cot_str)

    # Format the dataset
    multiplicand_rev = str(multiplicand)[::-1]
    multiplier_rev = str(multiplier)[::-1]
    q_str = ' * '.join([' '.join(list(multiplicand_rev)), ' '.join(list(multiplier_rev))])
    a_str = ' '.join(list(cumulative_sums[-1]))
    all_str = f"{q_str}||{cot_str} #### {a_str}"
    
    return all_str

if __name__ == '__main__':
    # Generate 808k training examples and 1k validation examples
    num_train = 808 * 1e3 # 1e3
    num_valid = 1 * 1e3 # 1e3
    num_test = 1 * 1e3 # 1e3

    for order in [4, 5, 6, 7, 8, 9, 10]:
        data_dir = f'data/{order}_by_{order}_mult'

        train_set = set()
        valid_set = set()
        test_set = set()

        # Generate training data
        while len(train_set) < num_train:
            example = generate_n_by_n(order)
            equation = example.split('\n')[0]  # Use the first line as a unique identifier
            if equation not in train_set:
                train_set.add(example)

        # Generate validation data
        while len(valid_set) < num_valid:
            example = generate_n_by_n(order)
            equation = example.split('\n')[0]
            if equation not in train_set and equation not in valid_set:
                valid_set.add(example)

        # Generate test data
        while len(test_set) < num_test:
            example = generate_n_by_n(order)
            equation = example.split('\n')[0]
            if equation not in train_set and equation not in valid_set and equation not in test_set:
                test_set.add(example)

        os.makedirs(data_dir, exist_ok=True)

        # Write training data to a file
        with open(f'{data_dir}/train.txt', 'w') as f:
            for example in train_set:
                f.write(example + '\n')

        # Write validation data to a file
        with open(f'{data_dir}/valid.txt', 'w') as f:
            for example in valid_set:
                f.write(example + '\n')

        # Write test data to a file
        with open(f'{data_dir}/test_bigbench.txt', 'w') as f:
            for example in test_set:
                f.write(example + '\n')
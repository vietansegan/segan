/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package sampling.util;

import java.util.ArrayList;

/**
 * Node object used in nested CRP
 *
 * @author vietan
 */
public class NCRPNode<N extends NCRPNode, C, U> extends Node<N, C> {

    protected ArrayList<U> customers; // set of customers assigned to this node
    protected int numPathCustomers; // number of customers assigned to any nodes in the subtree rooted at this node

    public NCRPNode(int index, int level, C content, N parent) {
        super(index, level, content, parent);
        this.customers = new ArrayList<U>();
        this.numPathCustomers = 0;
    }

    public ArrayList<U> getCustomers() {
        return this.customers;
    }

    public boolean isEmpty() {
        return this.getNumNodeCustomers() == 0;
    }

    /**
     * Add a customer to the current node
     *
     * @param customer The current customer to be added
     */
    public void addCustomer(U customer) {
        this.customers.add(customer);
        this.increaseNumPathCustomers(1);
    }

    /**
     * Increase the number of customers of the path from the root to this node
     *
     * @param count The number of customers to be added
     */
    public void increaseNumPathCustomers(int count) {
        NCRPNode<N, C, U> node = this;
        while (node != null) {
            node.numPathCustomers += count;
            node = node.getParent();
        }
    }

    /**
     * Remove a customer from the current node
     *
     * @param customer The customer to be removed
     */
    public void removeCustomer(U customer) {
        this.customers.remove(customer);
        if (this.getNumNodeCustomers() < 0) {
            throw new RuntimeException("Negative number of customers");
        }
        this.decreaseNumPathCustomers(1);
    }

    /**
     * Decrease the number of customers of the path from the root to this node
     *
     * @param count The number of customers to be decreased
     */
    public void decreaseNumPathCustomers(int count) {
        NCRPNode<N, C, U> node = this;
        while (node != null) {
            node.numPathCustomers -= count;
            if (node.numPathCustomers < 0) {
                throw new RuntimeException("Negative number of customers");
            }
            node = node.getParent();
        }
    }

    public int getNumNodeCustomers() {
        return this.customers.size();
    }

    public int getNumPathCustomers() {
        return numPathCustomers;
    }

    public void validate(String str) {
        int sumChildrentPathNumCustomers = 0;
        for (NCRPNode child : this.getChildren()) {
            sumChildrentPathNumCustomers += child.getNumPathCustomers();
        }
        if (sumChildrentPathNumCustomers + this.getNumNodeCustomers() != this.numPathCustomers) {
            throw new RuntimeException(str + ". Numbers of customers mismatch. "
                    + (sumChildrentPathNumCustomers + this.getNumNodeCustomers())
                    + " vs. " + numPathCustomers
                    + ". " + this.toString());
        }

        if (this.numPathCustomers < this.getNumNodeCustomers()) {
            throw new RuntimeException(str + ". Invalid number of customers");
        }

        if (this.isEmpty()) {
            throw new RuntimeException(str + ". Empty node: " + this.toString());
        }
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("[")
                .append(getPathString())
                .append(", #ch = ").append(getChildren().size())
                .append(", #n = ").append(getNumNodeCustomers())
                .append(", #p = ").append(getNumPathCustomers())
                .append("]");
        return str.toString();
    }
}
